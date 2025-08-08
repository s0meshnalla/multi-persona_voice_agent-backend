from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_session import Session
import asyncio
import threading
import uuid
import logging
from datetime import datetime
import json

# Import your enhanced panel system
from panel_system import IntelligentPanelSystem
from config import config

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'

# CORS configuration for React frontend
CORS(app, supports_credentials=True, origins=config.CORS_ORIGINS)
Session(app)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global storage for panel systems and conversations
panel_systems = {}  # session_id: IntelligentPanelSystem instances
conversations = {}  # session_id: conversation data

# Asyncio event loop for handling async panel operations
loop = asyncio.new_event_loop()

def start_background_loop(loop):
    """Start the background asyncio loop"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start background thread for async operations
thread = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
thread.start()

@app.route('/api/start-conversation', methods=['POST'])
def start_conversation():
    """Initialize a new enhanced conversation session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Create new enhanced panel system for this session
        panel_system = IntelligentPanelSystem()
        panel_systems[session_id] = panel_system
        
        # Initialize conversation history with enhanced tracking
        conversations[session_id] = {
            'messages': [],
            'agents_status': {
                'realist': {'active': True, 'confidence': 0},
                'optimist': {'active': True, 'confidence': 0},
                'expert': {'active': True, 'confidence': 0}
            },
            'active': True,
            'created_at': datetime.now().isoformat(),
            'user_context': {},
            'search_history': [],
            'handoff_count': 0
        }
        
        logger.info(f"Started new enhanced conversation: {session_id}")
        
        welcome_message = """Welcome to our AI panel discussion! 

I'm here with my colleagues - the **Realist**, **Optimist**, and **Expert**. We work together dynamically to give you the best advice:

• **Optimist Agent** - Finds opportunities and encouraging perspectives
• **Realist Agent** - Provides practical guidance and asks clarifying questions  
• **Expert Agent** - Offers current data and industry insights

The most relevant expert will respond first, and others will join when their expertise adds value. You can also click any panel member above to have them respond specifically.

What would you like us to explore together?"""
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Enhanced panel discussion started!',
            'welcome_message': welcome_message
        })
        
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send-message', methods=['POST', 'OPTIONS'])
def send_message():
    """Process user message with enhanced multi-agent coordination"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
        
    try:
        data = request.get_json()
        
        # Validate request data
        if not data:
            logger.error("No JSON data provided in send-message request")
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        user_message = data.get('message', '').strip()
        
        # Validate session
        if not session_id:
            logger.error("Session ID missing in send-message request")
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
            
        if session_id not in panel_systems:
            logger.error(f"Invalid session ID in send-message request: {session_id}")
            return jsonify({'success': False, 'error': 'Invalid or expired session. Please refresh the page.'}), 400
        
        # Validate message
        if not user_message:
            logger.error("Empty message in send-message request")
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
        
        logger.info(f"Processing message for session {session_id}: {user_message[:50]}...")
        
        # Add user message to conversation history
        conversations[session_id]['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process message with enhanced system
        async def process_enhanced_message():
            try:
                panel_system = panel_systems[session_id]
                response = await panel_system.process_user_input(user_message)
                return response
                
            except Exception as e:
                logger.error(f"Error in enhanced processing: {e}")
                return {
                    'content': 'Our panel is experiencing technical difficulties. Please try again.',
                    'agents': ['system'],
                    'confidence': {'system': 0.5},
                    'response_complete': True,
                    'used_search': False,
                    'turn_id': 0,
                    'agent_name': 'System'
                }
        
        # Execute async processing with increased timeout for search operations
        future = asyncio.run_coroutine_threadsafe(process_enhanced_message(), loop)
        panel_response = future.result(timeout=45)
        
        # Add enhanced panel response to conversation
        enhanced_message = {
            'role': 'panel',
            'content': panel_response['content'],
            'agents': panel_response['agents'],
            'confidence': panel_response['confidence'],
            'timestamp': datetime.now().isoformat(),
            'turn_id': panel_response.get('turn_id'),
            'used_search': panel_response.get('used_search', False),
            'search_info': panel_response.get('search_info', {}),
            'user_context_referenced': panel_response.get('user_context_referenced', []),
            'agent_name': panel_response.get('agent_name', 'Panel')
        }
        
        conversations[session_id]['messages'].append(enhanced_message)
        
        # Update agent status tracking
        conversations[session_id]['agents_status'].update({
            agent: {'active': True, 'confidence': conf} 
            for agent, conf in panel_response['confidence'].items()
        })
        
        # Track search usage
        if panel_response.get('used_search'):
            conversations[session_id]['search_history'].append({
                'timestamp': datetime.now().isoformat(),
                'agent': panel_response['agents'][0] if panel_response['agents'] else 'unknown',
                'query': panel_response.get('search_info', {}).get('query', ''),
                'turn_id': panel_response.get('turn_id')
            })
        
        # Update user context if new information was gathered
        if panel_response.get('user_context_referenced'):
            panel_system = panel_systems[session_id]
            conversations[session_id]['user_context'] = panel_system.memory_system.get_shared_context()
        
        return jsonify({
            'success': True,
            'response': panel_response['content'],
            'agents': panel_response['agents'],
            'confidence': panel_response['confidence'],
            'needs_handoff': panel_response.get('needs_handoff', False),
            'handoff_agent': panel_response.get('handoff_agent'),
            'handoff_reason': panel_response.get('handoff_reason', ''),
            'turn_id': panel_response.get('turn_id'),
            'response_complete': panel_response.get('response_complete', True),
            'used_search': panel_response.get('used_search', False),
            'search_info': panel_response.get('search_info', {}),
            'user_context_referenced': panel_response.get('user_context_referenced', []),
            'agent_name': panel_response.get('agent_name', 'Panel'),
            'streaming_text': panel_response.get('streaming_text', [])
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced message processing: {e}", exc_info=True)
        return jsonify({
            'success': False, 
            'error': 'Failed to process message',
            'details': str(e) if getattr(config, 'DEBUG', False) else 'Internal server error'
        }), 500

@app.route('/api/select-agent', methods=['POST', 'OPTIONS'])
def select_agent():
    """Allow manual agent selection from frontend"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200
        
    try:
        data = request.get_json()
        
        # Validate request data
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        session_id = data.get('session_id')
        selected_agent = data.get('agent')  # 'realist', 'optimist', 'expert'
        
        # Validate session
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
            
        if session_id not in panel_systems:
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
            
        # Validate agent
        if not selected_agent:
            return jsonify({'success': False, 'error': 'Agent name required'}), 400
            
        if selected_agent not in ['realist', 'optimist', 'expert']:
            return jsonify({'success': False, 'error': f'Invalid agent selected: {selected_agent}. Must be one of: realist, optimist, expert'}), 400
        
        # Set manual selection for this session
        panel_systems[session_id].set_manual_agent(selected_agent)
        
        logger.info(f"Agent {selected_agent} manually selected for session {session_id}")
        
        return jsonify({
            'success': True, 
            'selected_agent': selected_agent,
            'message': f'{selected_agent.title()} Agent will respond to your next message'
        })
        
    except Exception as e:
        logger.error(f"Error selecting agent: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/continue-conversation', methods=['POST'])
def continue_conversation():
    """Handle enhanced agent handoffs with full context"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        turn_id = data.get('turn_id')
        
        if not session_id or session_id not in panel_systems:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        if not turn_id:
            return jsonify({'success': False, 'error': 'Turn ID required'}), 400
        
        # Process handoff with enhanced system
        async def process_enhanced_handoff():
            try:
                panel_system = panel_systems[session_id]
                response = await panel_system.process_handoff(turn_id)
                return response
                
            except Exception as e:
                logger.error(f"Error in enhanced handoff processing: {e}")
                return {
                    'content': '',
                    'agents': [],
                    'confidence': {},
                    'response_complete': True,
                    'error': str(e),
                    'used_search': False,
                    'agent_name': 'System'
                }
        
        # Execute async handoff processing
        future = asyncio.run_coroutine_threadsafe(process_enhanced_handoff(), loop)
        handoff_response = future.result(timeout=30)
        
        if 'error' not in handoff_response and handoff_response['content']:
            # Add handoff response to conversation
            enhanced_handoff_message = {
                'role': 'panel',
                'content': handoff_response['content'],
                'agents': handoff_response['agents'],
                'confidence': handoff_response['confidence'],
                'timestamp': datetime.now().isoformat(),
                'turn_id': handoff_response.get('turn_id'),
                'is_handoff': True,
                'used_search': handoff_response.get('used_search', False),
                'search_info': handoff_response.get('search_info', {}),
                'user_context_referenced': handoff_response.get('user_context_referenced', []),
                'agent_name': handoff_response.get('agent_name', 'Panel')
            }
            
            conversations[session_id]['messages'].append(enhanced_handoff_message)
            conversations[session_id]['handoff_count'] += 1
            
            # Track search in handoff
            if handoff_response.get('used_search'):
                conversations[session_id]['search_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'agent': handoff_response['agents'][0] if handoff_response['agents'] else 'unknown',
                    'query': handoff_response.get('search_info', {}).get('query', ''),
                    'turn_id': handoff_response.get('turn_id'),
                    'is_handoff': True
                })
        
        return jsonify({
            'success': True,
            'response': handoff_response['content'],
            'agents': handoff_response['agents'],
            'confidence': handoff_response['confidence'],
            'needs_handoff': handoff_response.get('needs_handoff', False),
            'handoff_agent': handoff_response.get('handoff_agent'),
            'response_complete': handoff_response.get('response_complete', True),
            'used_search': handoff_response.get('used_search', False),
            'search_info': handoff_response.get('search_info', {}),
            'user_context_referenced': handoff_response.get('user_context_referenced', []),
            'agent_name': handoff_response.get('agent_name', 'Panel'),
            'streaming_text': handoff_response.get('streaming_text', [])
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced continue conversation: {e}")
        return jsonify({'success': False, 'error': 'Failed to continue conversation'}), 500

@app.route('/api/conversation-history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get full enhanced conversation history"""
    try:
        if session_id not in conversations:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        conversation_data = conversations[session_id]
        
        return jsonify({
            'success': True,
            'messages': conversation_data['messages'],
            'agents_status': conversation_data['agents_status'],
            'user_context': conversation_data.get('user_context', {}),
            'search_history': conversation_data.get('search_history', []),
            'conversation_stats': {
                'total_messages': len(conversation_data['messages']),
                'handoff_count': conversation_data.get('handoff_count', 0),
                'search_count': len(conversation_data.get('search_history', [])),
                'created_at': conversation_data['created_at']
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting enhanced history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-user-context', methods=['GET'])
def get_user_context():
    """Get stored user context for enhanced personalization"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in panel_systems:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        panel_system = panel_systems[session_id]
        shared_context = panel_system.memory_system.get_shared_context()
        
        # Get recent context references
        recent_messages = conversations[session_id]['messages'][-5:]
        recent_references = []
        for msg in recent_messages:
            if msg.get('user_context_referenced'):
                recent_references.extend(msg['user_context_referenced'])
        
        return jsonify({
            'success': True,
            'user_context': shared_context,
            'context_keys': list(shared_context.keys()),
            'recent_references': list(set(recent_references)),
            'context_summary': {
                'total_facts': len(shared_context),
                'last_updated': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search-info', methods=['POST'])
def get_search_info():
    """Get enhanced information about recent searches and data usage"""
    try:
        session_id = request.get_json().get('session_id')
        
        if not session_id or session_id not in conversations:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        search_history = conversations[session_id].get('search_history', [])
        
        # Analyze search patterns
        agent_search_counts = {}
        total_searches = len(search_history)
        
        for search in search_history:
            agent = search.get('agent', 'unknown')
            agent_search_counts[agent] = agent_search_counts.get(agent, 0) + 1
        
        # Find most active agent
        most_active_agent = None
        if agent_search_counts:
            most_active_agent = sorted(agent_search_counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        return jsonify({
            'success': True,
            'search_history': search_history,
            'search_analytics': {
                'total_searches': total_searches,
                'agent_breakdown': agent_search_counts,
                'most_active_agent': most_active_agent,
                'handoff_searches': len([s for s in search_history if s.get('is_handoff', False)])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting search info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conversation-summary', methods=['POST'])
def get_conversation_summary():
    """Generate enhanced conversation summary with key insights"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in conversations:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        conversation_data = conversations[session_id]
        messages = conversation_data['messages']
        
        # Generate summary insights
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        panel_messages = [msg for msg in messages if msg['role'] == 'panel']
        
        # Analyze conversation characteristics
        topics_discussed = set()
        agents_participated = set()
        search_queries = []
        
        for msg in panel_messages:
            if msg.get('agents'):
                agents_participated.update(msg['agents'])
            if msg.get('search_info') and msg['search_info'].get('query'):
                search_queries.append(msg['search_info']['query'])
        
        # Extract key decision points
        key_decisions = []
        for msg in user_messages:
            if any(word in msg['content'].lower() for word in ['should', 'decision', 'choose', 'decide']):
                key_decisions.append(msg['content'][:100] + "...")
        
        summary = {
            'conversation_overview': {
                'total_exchanges': len(user_messages),
                'agents_participated': list(agents_participated),
                'search_queries_made': len(search_queries),
                'duration': conversation_data['created_at']
            },
            'key_topics': {
                'decisions_discussed': key_decisions[:3],  # Top 3
                'search_areas': search_queries[:5],  # Recent 5
                'user_context_discovered': conversation_data.get('user_context', {})
            },
            'panel_insights': {
                'most_active_agent': max(agents_participated, key=lambda x: len([m for m in panel_messages if x in m.get('agents', [])])) if agents_participated else None,
                'handoffs_performed': conversation_data.get('handoff_count', 0),
                'data_searches_performed': len([m for m in panel_messages if m.get('used_search', False)])
            }
        }
        
        return jsonify({
            'success': True,
            'conversation_summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error generating conversation summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/end-conversation', methods=['POST'])
def end_conversation():
    """End enhanced conversation with cleanup and summary"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id in panel_systems:
            # Clean up panel system resources
            del panel_systems[session_id]
        
        if session_id in conversations:
            conversations[session_id]['active'] = False
            conversations[session_id]['ended_at'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True, 
            'message': 'Enhanced conversation ended successfully'
        })
        
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with system statistics"""
    try:
        active_conversations = len([c for c in conversations.values() if c.get('active', False)])
        total_conversations = len(conversations)
        
        # Calculate system usage stats
        total_messages = sum(len(c.get('messages', [])) for c in conversations.values())
        total_searches = sum(len(c.get('search_history', [])) for c in conversations.values())
        total_handoffs = sum(c.get('handoff_count', 0) for c in conversations.values())
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_stats': {
                'active_conversations': active_conversations,
                'total_conversations': total_conversations,
                'total_messages_processed': total_messages,
                'total_searches_performed': total_searches,
                'total_handoffs_executed': total_handoffs
            },
            'features_enabled': {
                'web_search': bool(config.SERPER_API_KEY),
                'memory_system': True,
                'agent_handoffs': True,
                'voice_integration': True,
                'manual_agent_selection': True
            }
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/reset-context', methods=['POST'])
def reset_context():
    """Reset conversation context and memory for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in panel_systems:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        # Reset the panel system memory
        panel_systems[session_id].memory_system = panel_systems[session_id].__class__().memory_system
        
        # Reset conversation data
        conversations[session_id]['user_context'] = {}
        conversations[session_id]['search_history'] = []
        conversations[session_id]['handoff_count'] = 0
        
        return jsonify({
            'success': True,
            'message': 'Context and memory reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting context: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Enhanced 404 handler"""
    return jsonify({
        'success': False,
        'error': 'API endpoint not found',
        'available_endpoints': [
            'POST /api/start-conversation',
            'POST /api/send-message', 
            'POST /api/select-agent',
            'POST /api/continue-conversation',
            'GET /api/conversation-history/<session_id>',
            'GET /api/get-user-context',
            'POST /api/search-info',
            'POST /api/conversation-summary',
            'POST /api/end-conversation',
            'POST /api/reset-context',
            'GET /api/health'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Enhanced 500 handler"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Our enhanced panel system encountered an issue. Please try again.'
    }), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced AI Panel Discussion Server...")
    logger.info(f"Features enabled: Web Search={bool(getattr(config, 'SERPER_API_KEY', False))}, Voice=True, Memory=True")
    
    try:
        # Start the application
        app.run(
            debug=getattr(config, 'DEBUG', True),
            host=getattr(config, 'HOST', '0.0.0.0'),
            port=getattr(config, 'PORT', 5000),
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

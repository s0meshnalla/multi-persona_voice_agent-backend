import asyncio
import logging
import json
import re
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import config

logger = logging.getLogger(__name__)

# ===== ENHANCED MEMORY SYSTEM WITH PERSISTENT CONTEXT =====
class AdvancedMemorySystem:
    def __init__(self):
        self.shared_context = {}  # User facts across conversation
        self.conversation_memory = []  # Full conversation history
        self.user_profile = {}  # Persistent user profile
        logger.info("✅ Enhanced memory system initialized")
    
    def store_memory(self, content: str, memory_type: str, metadata: Dict[str, Any] = None):
        """Store memory with enhanced user context extraction"""
        try:
            user_facts = self._extract_comprehensive_user_facts(content)
            self.shared_context.update(user_facts)
            
            # Update user profile
            if user_facts:
                self.user_profile.update(user_facts)
            
            memory_entry = {
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                "user_facts": user_facts,
                "turn_context": len(self.conversation_memory)
            }
            
            if metadata:
                memory_entry.update(metadata)
                
            self.conversation_memory.append(memory_entry)
            
            # Keep extensive memory (up to 100 entries)
            if len(self.conversation_memory) > 100:
                self.conversation_memory = self.conversation_memory[-100:]
            
            if user_facts:
                logger.info(f"📝 Stored user context: {user_facts}")
                
        except Exception as e:
            logger.error(f"Memory storage error: {e}")
    
    def get_shared_context(self) -> Dict[str, Any]:
        """Get comprehensive shared context"""
        return {**self.shared_context, **self.user_profile}
    
    def retrieve_relevant_memories(self, query: str, n_results: int = 8, memory_types: List[str] = None):
        """Enhanced memory retrieval with better context"""
        try:
            query_lower = query.lower()
            relevant = []
            
            # Search through all conversation memory
            for memory in self.conversation_memory:
                content_lower = memory["content"].lower()
                relevance_score = 0
                
                # Calculate relevance based on keyword matching
                query_words = query_lower.split()
                for word in query_words:
                    if word in content_lower:
                        relevance_score += 1
                
                # Boost recent memories
                recency_boost = max(0, 1 - (len(self.conversation_memory) - memory.get("turn_context", 0)) / 50)
                relevance_score += recency_boost
                
                if relevance_score > 0:
                    relevant.append({
                        "content": memory["content"],
                        "metadata": {"type": memory["type"], "timestamp": memory["timestamp"]},
                        "user_facts": memory.get("user_facts", {}),
                        "relevance": relevance_score
                    })
            
            # Sort by relevance and return top results
            relevant.sort(key=lambda x: x["relevance"], reverse=True)
            return relevant[:n_results]
            
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []
    
    def get_conversation_context(self, last_n: int = 10) -> str:
        """Get recent conversation context"""
        recent_memories = self.conversation_memory[-last_n:]
        context_lines = []
        
        for memory in recent_memories:
            if memory["type"] in ["user_input", "agent_response"]:
                content = memory["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                context_lines.append(content)
        
        return "\n".join(context_lines)
    
    def _extract_comprehensive_user_facts(self, content: str) -> Dict[str, str]:
        """Enhanced user fact extraction matching example scenarios"""
        facts = {}
        content_lower = content.lower()
        
        # Extract field of study / major
        study_patterns = [
            r"(?:studying|study|major in|majoring in|i'm in)\s+([a-zA-Z\s]+?)(?:\s+and|\s*,|\s*\.|$)",
            r"(?:I'm|I am)\s+(?:a\s+)?([a-zA-Z\s]+?)\s+(?:student|major)",
            r"my major is\s+([a-zA-Z\s]+)",
            r"computer science|data science|engineering|business|psychology|biology|chemistry|physics"
        ]
        
        for pattern in study_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if pattern == r"computer science|data science|engineering|business|psychology|biology|chemistry|physics":
                    facts["major"] = match.group(0)
                else:
                    major = match.group(1).strip()
                    if 2 < len(major) < 30:
                        facts["major"] = major
                break
        
        # Extract project/interest details
        project_patterns = [
            r"(?:project|working on|building|developing)\s+(?:is\s+)?(?:on\s+)?([^,.!?]+)",
            r"(?:interested in|want to learn|focus on)\s+([^,.!?]+)"
        ]
        
        for pattern in project_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                project = match.group(1).strip()
                if 3 < len(project) < 50:
                    facts["project_interest"] = project
                    break
        
        # Extract budget/financial info
        budget_match = re.search(r"(?:budget|afford|support|family can support)\s+(?:about\s+)?\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", content)
        if budget_match:
            facts["budget"] = budget_match.group(1)
        
        # Extract technology preferences
        tech_keywords = ["react", "angular", "python", "javascript", "machine learning", "ml", "ai", "web development"]
        for tech in tech_keywords:
            if tech in content_lower:
                if "technologies" not in facts:
                    facts["technologies"] = []
                facts["technologies"].append(tech)
        
        # Extract timeline concerns
        timeline_patterns = [
            r"(?:time management|timeline|how long|when to)",
            r"(?:final year|graduating|last semester)"
        ]
        
        for pattern in timeline_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                facts["timeline_concern"] = "yes"
                break
        
        return facts

# ===== ENHANCED WEB SEARCH TOOL =====
class ProactiveWebSearchInput(BaseModel):
    query: str = Field(description="The search query string")

class ProactiveWebSearchTool(BaseTool):
    name: str = "proactive_web_search"
    description: str = "Search for current information"
    args_schema: type[BaseModel] = ProactiveWebSearchInput
    
    def __init__(self):
        super().__init__()
        self._api_key = config.SERPER_API_KEY
    
    def _run(self, query: str) -> str:
        return self.search_for_context(query)
    
    async def _arun(self, query: str) -> str:
        return self.search_for_context(query)
    
    def search_for_context(self, query: str, context: str = "") -> str:
        """Enhanced search with better result processing"""
        try:
            if not self._api_key:
                return json.dumps({"error": "No search API configured", "success": False})
            
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json"
            }
            
            # Enhanced query with current year
            enhanced_query = f"{query} 2024"
            
            payload = {
                "q": enhanced_query,
                "num": 5,
                "hl": "en",
                "gl": "us"
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            results = []
            
            # Process featured snippets first
            if "answerBox" in search_data:
                answer = search_data["answerBox"]
                results.append({
                    "type": "featured",
                    "content": answer.get("snippet", ""),
                    "source": answer.get("title", ""),
                    "priority": "high"
                })
            
            # Process organic results
            if "organic" in search_data:
                for item in search_data["organic"][:3]:
                    results.append({
                        "type": "organic",
                        "content": item.get("snippet", ""),
                        "title": item.get("title", ""),
                        "priority": "medium"
                    })
            
            return json.dumps({
                "query": enhanced_query,
                "results": results,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return json.dumps({"error": str(e), "success": False})

# ===== ENHANCED AGENT CLASSES MATCHING EXAMPLE SCENARIOS =====
class EnhancedBasePanelAgent:
    def __init__(self, role: str, temperature: float = 0.7):
        self.role = role
        self.model = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=150
        )
        self.search_tool = ProactiveWebSearchTool()
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        return f"""You are the {self.role} Agent in an AI panel discussion.

CRITICAL BEHAVIOR (matching examples):
1. ALWAYS start with "{self.role} Agent:" (exactly like the examples)
2. Use search proactively when discussing current trends, job markets, statistics
3. Reference stored user information: "Since you mentioned [detail from memory]..."
4. Store important user details: [Agent stores in vector memory: "User: detail"]
5. Ask clarifying questions when needed: "What's your major?" or "Have you thought about...?"
6. Suggest handoffs naturally: "Let me have the [Agent] add their perspective on..."

{self._get_specific_instructions()}

RESPONSE FORMAT (80-120 words):
- Start with clear identification: "{self.role} Agent:"
- Provide your expertise-based perspective
- Use search data when relevant: "Recent studies show..." or "Current data indicates..."
- Reference user context when available
- End with handoff suggestion if another agent would help"""
    
    def _get_specific_instructions(self) -> str:
        return "Provide expert advice in your area."

class OptimistAgent(EnhancedBasePanelAgent):
    def __init__(self):
        super().__init__("Optimist", temperature=0.7)
    
    def _get_specific_instructions(self) -> str:
        return """OPTIMIST AGENT EXPERTISE:
- Find encouraging data and success stories
- Highlight opportunities and benefits
- Provide motivation while being realistic
- Search for positive statistics: "internship benefits", "study abroad advantages"
- Focus on potential and growth opportunities
- Use phrases like "That's exciting!" or "Great opportunity!" 
- Always find the positive angle while being helpful"""

class RealistAgent(EnhancedBasePanelAgent):
    def __init__(self):
        super().__init__("Realist", temperature=0.3)
    
    def _get_specific_instructions(self) -> str:
        return """REALIST AGENT EXPERTISE:
- Practical planning and implementation
- Ask clarifying questions about specifics: major, timeline, budget
- Store user details carefully for future reference
- Focus on realistic timelines and practical steps
- Consider constraints and challenges
- Search for practical information: costs, requirements, timelines
- Use phrases like "Let's consider..." or "Here's what you need to know..."
- Balance optimism with practical reality"""

class ExpertAgent(EnhancedBasePanelAgent):
    def __init__(self):
        super().__init__("Expert", temperature=0.4)
    
    def _get_specific_instructions(self) -> str:
        return """EXPERT AGENT EXPERTISE:
- Always search for current data and statistics
- Lead with facts: "Current data shows..." or "Recent analysis reveals..."
- Provide industry insights and market analysis
- Use specific numbers and percentages from search results
- Focus on technical and analytical information
- Search for: job market data, industry trends, employment statistics
- Reference authoritative sources naturally
- Make complex information accessible"""

# ===== MAIN ENHANCED PANEL SYSTEM =====
class IntelligentPanelSystem:
    def __init__(self):
        self.memory_system = AdvancedMemorySystem()
        self.agents = {
            "optimist": OptimistAgent(),
            "realist": RealistAgent(),
            "expert": ExpertAgent()
        }
        self.conversation_history = []
        self.current_turn = 0
        self.pending_handoffs = {}
        self.manual_agent_selection = None  # For frontend agent selection
        self.last_speaker = None
    
    def set_manual_agent(self, agent_name: str):
        """Allow manual agent selection from frontend"""
        if agent_name in self.agents:
            self.manual_agent_selection = agent_name
            logger.info(f"🎯 Manual agent selection: {agent_name}")
    
    async def process_user_input(self, user_input: str) -> Dict:
        """Enhanced processing matching example scenarios"""
        try:
            self.current_turn += 1
            
            # Store user input with enhanced context
            self.memory_system.store_memory(user_input, "user_input")
            self.conversation_history.append(f"User: {user_input}")
            
            # Select agent (manual override or intelligent selection)
            if self.manual_agent_selection:
                primary_agent = self.manual_agent_selection
                self.manual_agent_selection = None  # Clear after use
                logger.info(f"🎯 Using manually selected agent: {primary_agent}")
            else:
                primary_agent = self._select_agent_scenario_based(user_input)
                logger.info(f"🎯 Intelligently selected agent: {primary_agent}")
            
            # Generate enhanced response
            response_data = await self._generate_scenario_based_response(
                primary_agent, user_input, is_primary=True
            )
            
            # Enhanced handoff logic
            handoff_decision = self._check_scenario_handoff(
                user_input, response_data['content'], primary_agent
            )
            
            # Store handoff if needed
            if handoff_decision['needs_handoff']:
                self.pending_handoffs[self.current_turn] = {
                    'handoff_agent': handoff_decision['handoff_agent'],
                    'reason': handoff_decision['reason'],
                    'user_input': user_input,
                    'primary_agent': primary_agent
                }
            
            # Store response in memory
            self.memory_system.store_memory(
                response_data['content'], 
                "agent_response", 
                {"agent": primary_agent, "turn": self.current_turn}
            )
            
            self.conversation_history.append(f"{primary_agent.title()} Agent: {response_data['content']}")
            self.last_speaker = primary_agent
            
            return {
                'content': response_data['content'],
                'agents': [primary_agent],
                'confidence': {primary_agent: 0.9},
                'needs_handoff': handoff_decision['needs_handoff'],
                'handoff_agent': handoff_decision.get('handoff_agent'),
                'handoff_reason': handoff_decision.get('reason'),
                'turn_id': self.current_turn,
                'response_complete': not handoff_decision['needs_handoff'],
                'used_search': response_data.get('used_search', False),
                'agent_name': f"{primary_agent.title()} Agent",
                'streaming_text': response_data.get('streaming_text', [])
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return self._create_error_response()
    
    def _select_agent_scenario_based(self, user_input: str) -> str:
        """Agent selection matching example scenarios"""
        input_lower = user_input.lower()
        shared_context = self.memory_system.get_shared_context()
        
        # Optimist leads on: decisions, opportunities, encouraging topics
        optimist_triggers = [
            'should i', 'thinking about', 'considering', 'want to',
            'excited about', 'opportunity', 'dream', 'goal'
        ]
        if any(trigger in input_lower for trigger in optimist_triggers):
            return "optimist"
        
        # Expert leads on: technical questions, current trends, data requests
        expert_triggers = [
            'vs', 'better', 'latest', 'current', 'trend', 'demand',
            'job market', 'which framework', 'technology', 'data'
        ]
        if any(trigger in input_lower for trigger in expert_triggers):
            return "expert"
        
        # Realist leads on: practical questions, planning, constraints
        realist_triggers = [
            'how long', 'timeline', 'practical', 'budget', 'cost',
            'worried about', 'time management', 'realistic'
        ]
        if any(trigger in input_lower for trigger in realist_triggers):
            return "realist"
        
        # Default: Optimist for general questions (matches examples)
        return "optimist"
    
    async def _generate_scenario_based_response(self, agent_name: str, user_input: str, is_primary: bool = True) -> Dict:
        """Generate responses matching example scenarios"""
        try:
            agent = self.agents[agent_name]
            shared_context = self.memory_system.get_shared_context()
            
            # Proactive search based on agent and query
            should_search = self._should_search_proactively(user_input, agent_name)
            search_results = {}
            search_query = ""
            
            if should_search:
                search_query = self._generate_contextual_search_query(user_input, agent_name, shared_context)
                logger.info(f"🔍 {agent_name.title()} Agent searching: {search_query}")
                search_results_str = agent.search_tool.search_for_context(search_query)
                try:
                    search_results = json.loads(search_results_str)
                except:
                    search_results = {"success": False}
            
            # Build comprehensive context
            context = self._build_scenario_context(
                user_input, agent_name, shared_context, search_results, search_query
            )
            
            # Generate response
            messages = [
                SystemMessage(content=agent.system_prompt),
                HumanMessage(content=context)
            ]
            
            response = await agent.model.ainvoke(messages)
            content = response.content.strip()
            
            # Ensure proper agent identification
            if not content.startswith(f"{agent_name.title()} Agent:"):
                content = f"{agent_name.title()} Agent: {content}"
            
            return {
                'content': content,
                'used_search': should_search and search_results.get('success', False),
                'search_query': search_query if should_search else None,
                'streaming_text': self._create_streaming_chunks(content)
            }
            
        except Exception as e:
            logger.error(f"Error generating response for {agent_name}: {e}")
            return {
                'content': f"{agent_name.title()} Agent: Let me help you with that.",
                'used_search': False,
                'streaming_text': []
            }
    
    def _should_search_proactively(self, user_input: str, agent_name: str) -> bool:
        """Proactive search matching example scenarios"""
        input_lower = user_input.lower()
        
        # Search triggers for each agent type
        search_indicators = {
            "optimist": [
                "internship", "study abroad", "opportunity", "benefits",
                "should i", "thinking about", "career prospects"
            ],
            "expert": [
                "react vs angular", "job market", "demand", "current",
                "latest", "trend", "statistics", "data", "vs", "better"
            ],
            "realist": [
                "cost", "budget", "timeline", "requirement", "practical",
                "how long", "affordable", "visa", "process"
            ]
        }
        
        agent_triggers = search_indicators.get(agent_name, [])
        return any(trigger in input_lower for trigger in agent_triggers)
    
    def _generate_contextual_search_query(self, user_input: str, agent_name: str, shared_context: Dict) -> str:
        """Generate search queries matching example scenarios"""
        input_lower = user_input.lower()
        
        # Add user context to search
        context_additions = []
        if shared_context.get("major"):
            context_additions.append(shared_context["major"])
        
        # Agent-specific search query generation
        if agent_name == "optimist":
            if "internship" in input_lower:
                return f"internship benefits career impact statistics 2024 {' '.join(context_additions)}"
            elif "study abroad" in input_lower:
                return f"study abroad benefits international students statistics 2024"
            else:
                return f"{user_input} benefits opportunities 2024 {' '.join(context_additions)}"
        
        elif agent_name == "expert":
            if "react" in input_lower and "angular" in input_lower:
                return "React vs Angular job market demand 2024"
            elif "job market" in input_lower:
                return f"job market demand employment statistics 2024 {' '.join(context_additions)}"
            else:
                return f"{user_input} market analysis data 2024 {' '.join(context_additions)}"
        
        elif agent_name == "realist":
            if "study abroad" in input_lower:
                return f"affordable data science masters programs international students {shared_context.get('budget', '')}"
            else:
                return f"{user_input} practical requirements cost timeline 2024"
        
        return f"{user_input} 2024 {' '.join(context_additions)}"
    
    def _build_scenario_context(self, user_input: str, agent_name: str, shared_context: Dict, search_results: Dict, search_query: str) -> str:
        """Build context matching example scenarios"""
        context_parts = [f"USER'S QUESTION: {user_input}"]
        
        # Add user background
        if shared_context:
            context_details = []
            for key, value in shared_context.items():
                if isinstance(value, list):
                    context_details.append(f"{key}: {', '.join(value)}")
                else:
                    context_details.append(f"{key}: {value}")
            context_parts.append(f"USER BACKGROUND: {'; '.join(context_details)}")
        
        # Add search results if available
        if search_results.get('success') and search_results.get('results'):
            search_info = []
            for result in search_results['results'][:2]:
                if result.get('content'):
                    search_info.append(f"- {result['content'][:150]}")
            
            if search_info:
                context_parts.append(f"CURRENT DATA (from search '{search_query}'):\n" + "\n".join(search_info))
        
        # Add recent conversation context
        recent_context = self.memory_system.get_conversation_context(6)
        if recent_context:
            context_parts.append(f"RECENT CONVERSATION:\n{recent_context}")
        
        # Agent-specific instructions
        context_parts.append(f"""
RESPOND AS {agent_name.upper()} AGENT (matching example scenarios):
- Start with clear identification: "{agent_name.title()} Agent:"
- Use search data naturally if you searched: "Recent studies show..." or "Current data indicates..."
- Store user details when relevant: [Agent stores in vector memory: "User: detail"]
- Reference user context: "Since you mentioned [detail]..." or "Based on your [background]..."
- Ask clarifying questions if needed: "What's your major?" or "Have you considered...?"
- Suggest handoff if helpful: "Let me have the [Agent] add their perspective..."
- Be conversational and engaging (80-120 words)""")
        
        return "\n\n".join(context_parts)
    
    def _check_scenario_handoff(self, user_input: str, response_content: str, primary_agent: str) -> Dict:
        """Enhanced handoff logic matching example scenarios"""
        input_lower = user_input.lower()
        
        # Multi-faceted questions that benefit from multiple perspectives
        if primary_agent == "optimist":
            # After encouraging user, hand off to Realist for practical details
            if any(word in response_content.lower() for word in ['great', 'exciting', 'opportunity', 'benefits']):
                if any(word in input_lower for word in ['should', 'decision', 'consider', 'thinking']):
                    return {
                        'needs_handoff': True,
                        'handoff_agent': 'realist',
                        'reason': 'User needs practical guidance after encouragement'
                    }
        
        elif primary_agent == "realist":
            # After asking for details, might hand off to Expert for data or back to user
            if "?" in response_content:  # Asked a question
                if any(word in input_lower for word in ['data', 'current', 'market', 'trends']):
                    return {
                        'needs_handoff': True,
                        'handoff_agent': 'expert',
                        'reason': 'User needs current data after practical discussion'
                    }
        
        elif primary_agent == "expert":
            # After providing data, hand off to Realist for implementation
            if any(word in response_content.lower() for word in ['data shows', 'statistics', 'research']):
                if any(word in input_lower for word in ['how', 'implement', 'start', 'practical']):
                    return {
                        'needs_handoff': True,
                        'handoff_agent': 'realist',
                        'reason': 'User needs practical implementation after data'
                    }
        
        return {'needs_handoff': False}
    
    def _create_streaming_chunks(self, content: str) -> List[str]:
        """Create streaming chunks for voice output"""
        # Split by sentences for natural speech pauses
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        for sentence in sentences:
            if sentence.strip():
                chunks.append(sentence.strip() + ' ')
        return chunks if chunks else [content]
    
    async def process_handoff(self, turn_id: int) -> Dict:
        """Process handoffs matching example scenarios"""
        try:
            handoff_info = self.pending_handoffs.get(turn_id)
            if not handoff_info:
                return self._create_empty_response("No pending handoff")
            
            handoff_agent = handoff_info['handoff_agent']
            user_input = handoff_info['user_input']
            
            # Generate handoff response
            response_data = await self._generate_handoff_response(
                handoff_agent, user_input, handoff_info['primary_agent']
            )
            
            # Store handoff response
            self.memory_system.store_memory(
                response_data['content'],
                "agent_response",
                {"agent": handoff_agent, "turn": turn_id, "type": "handoff"}
            )
            
            self.conversation_history.append(f"{handoff_agent.title()} Agent: {response_data['content']}")
            self.last_speaker = handoff_agent
            
            # Clear handoff
            if turn_id in self.pending_handoffs:
                del self.pending_handoffs[turn_id]
            
            return {
                'content': response_data['content'],
                'agents': [handoff_agent],
                'confidence': {handoff_agent: 0.85},
                'needs_handoff': False,  # Limit to one handoff per turn
                'turn_id': turn_id,
                'response_complete': True,
                'used_search': response_data.get('used_search', False),
                'agent_name': f"{handoff_agent.title()} Agent",
                'streaming_text': response_data.get('streaming_text', [])
            }
            
        except Exception as e:
            logger.error(f"Error in handoff: {e}")
            return self._create_empty_response(str(e))
    
    async def _generate_handoff_response(self, agent_name: str, user_input: str, previous_agent: str) -> Dict:
        """Generate handoff response"""
        try:
            agent = self.agents[agent_name]
            shared_context = self.memory_system.get_shared_context()
            
            # Check if handoff agent should search
            should_search = self._should_search_proactively(user_input, agent_name)
            search_results = {}
            search_query = ""
            
            if should_search:
                search_query = self._generate_contextual_search_query(user_input, agent_name, shared_context)
                logger.info(f"🔍 {agent_name.title()} Agent (handoff) searching: {search_query}")
                search_results_str = agent.search_tool.search_for_context(search_query)
                try:
                    search_results = json.loads(search_results_str)
                except:
                    search_results = {"success": False}
            
            # Build handoff context
            context = f"""HANDOFF SITUATION:
USER'S ORIGINAL QUESTION: {user_input}
PREVIOUS AGENT: The {previous_agent.title()} Agent has already responded.
USER BACKGROUND: {'; '.join([f'{k}: {v}' for k, v in shared_context.items()]) if shared_context else "None yet"}

{f"CURRENT DATA (from search '{search_query}'):\n" + chr(10).join([f"- {r.get('content', '')[:150]}" for r in search_results.get('results', [])[:2] if r.get('content')]) if search_results.get('success') else ""}

RECENT CONVERSATION:
{self.memory_system.get_conversation_context(4)}

RESPOND AS {agent_name.upper()} AGENT:
- Start with clear identification: "{agent_name.title()} Agent:"
- Build on what the {previous_agent.title()} Agent said
- Add your unique {agent_name} expertise
- Use search data naturally if you searched
- Reference user context when relevant
- Keep focused and valuable (60-100 words)"""

            messages = [
                SystemMessage(content=agent.system_prompt),
                HumanMessage(content=context)
            ]
            
            response = await agent.model.ainvoke(messages)
            content = response.content.strip()
            
            # Ensure proper agent identification
            if not content.startswith(f"{agent_name.title()} Agent:"):
                content = f"{agent_name.title()} Agent: {content}"
            
            return {
                'content': content,
                'used_search': should_search and search_results.get('success', False),
                'streaming_text': self._create_streaming_chunks(content)
            }
            
        except Exception as e:
            logger.error(f"Error in handoff response for {agent_name}: {e}")
            return {
                'content': f"{agent_name.title()} Agent: Let me add my perspective to help complete this discussion.",
                'used_search': False,
                'streaming_text': []
            }
    
    def _create_error_response(self) -> Dict:
        return {
            'content': 'Our panel is experiencing technical difficulties.',
            'agents': ['system'],
            'confidence': {'system': 0.5},
            'response_complete': True,
            'turn_id': self.current_turn
        }
    
    def _create_empty_response(self, error: str) -> Dict:
        return {
            'content': '',
            'agents': [],
            'confidence': {},
            'response_complete': True,
            'error': error
        }

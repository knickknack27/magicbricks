import streamlit as st
import json
import os
import requests
import logging
import re
# Add LangChain imports
from langchain.memory import ConversationBufferMemory
# Add PyPDF import

# If running locally, load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Load context data from output.txt
with open('output.txt', 'r', encoding='utf-8') as f:
    context_data = json.load(f)

def normalize_name(name):
    # Remove trailing dashes/spaces, replace multiple spaces/dashes with single space, lowercase
    name = re.sub(r'[-\s]+$', '', name)
    name = re.sub(r'[-\s]+', ' ', name)
    return name.lower().strip()

# Load property URLs from urls.txt as a mapping (robust parsing)
url_map = {}
with open('urls.txt', 'r', encoding='utf-8') as f:
    for line in f:
        url_match = re.search(r'(https?://\S+)', line)
        if url_match:
            url = url_match.group(1).strip()
            name_part = line[:url_match.start()].strip()
            name = normalize_name(name_part)
            url_map[name] = url

# Azure OpenAI config from environment variables
AZURE_GPT41_URI = os.getenv('AZURE_GPT41_URI')
AZURE_GPT41mini_URI = os.getenv('AZURE_GPT41mini_URI')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
#ZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15')

# Initialize conversation memory (in Streamlit session state)
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Add file handler for LLM outputs
llm_log_path = 'llm_outputs.log'
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) and llm_log_path in h.baseFilename for h in logger.handlers):
    llm_file_handler = logging.FileHandler(llm_log_path, mode='a', encoding='utf-8')
    llm_file_handler.setLevel(logging.INFO)
    llm_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(llm_file_handler)

def find_relevant_context(query, context_data, top_k=3):
    # Simple keyword search for demo; replace with embedding search for production
    scored = []
    for entry in context_data:
        content = entry.get('content', {})
        text = '\n'.join([f"{k}: {v}" for k, v in content.items()])
        score = sum([text.lower().count(word) for word in query.lower().split()])
        scored.append((score, entry))  # Return the whole entry, not just text
    # Sort by score only, ignoring the entry dictionary
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for score, entry in scored[:top_k] if score > 0]

def generate_rag_prompt(query, context_chunks, chat_history=None):
    context = '\n---\n'.join(context_chunks)
    history = ""
    if chat_history:
        # Format chat history for the prompt using .type and .content
        history = "\n".join([
            f"User: {msg.content}" if getattr(msg, 'type', None) == 'human' else f"Raj: {msg.content}" for msg in chat_history
        ])
        history = f"\nChat History:\n{history}\n"
    prompt = f"""Your name is Raj. You will speak in Hinglish language by default. You are a professional real estate sales agent for Magic Bricks. Your goal is to provide comparison between multiple properties to the users. You will receive a document containing data on multiple properties (address, price, size, rooms, amenities, year built, neighbourhood info, photos, etc.). Your task is to:
  
  1. Unless the user wants to know about a specific property, identify the user's preferences only regarding house configuration and budget to narrow down property selection from your database, but don't irritate them by repeatedly asking too many questions if you feel they don't have a lot of preference. 
     These will be conversational questions so the outputs need to be short. Also, you will not ask all these preference questions in a single output, you will ask them one by one while creating a casual and fun conversation.
  2. Extract the relevant fields from the provided document.
  3. After you have collected the user preferences, compose a short, engaging description in natural language that's:
     - Concise: Keep it under 5–7 sentences.
     - Descriptive: Highlight location, layout, standout features, and lifestyle benefits.
     - Accurate: Only state facts present in the data. If something isn't in the data, ask for clarification.
  4. Based on the user's input question, you should ask what they are looking for, for example, the budget, luxury style or other preferences, amenities they want, 2bhk/3bhk/4bhk etc.
  5. You will never generate long paragraph-like answers to user's queries.
  6. When listing property options, always use a numbered or bulleted list. Do not write a single long paragraph. Each property should be a separate point with only the most essential details (name, price range, possession date, 1-2 key amenities).
  7. Never repeat the user's preferences in your output.
  8. If the user asks for options, do not summarize or generalize; just list the options as per the above format.

Conversational guidelines:
1. You will not be repetitive in your responses.
2. You will be concise, fun and conversational.
3. You will not behave casually with the user, treat them with respect and be professional.
4. You will not repeat the user preferences in your output for example, "Great choice! Aapko 3 BHK chahiye within 2 crore, right?".

Warnings:
You will not fabricate data.
You will generate content for the responses strictly for the mentioned real estate properties if asked for.
You will only provide content to the responses that is available in the document provided to you. 
If the user wants to know the complete details of the property, you will prompt them to visit Magic Bricks.

You will be highly rewarded for following all the given instructions diligently.\n{history}\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    return prompt

def get_llm_response(prompt, target_uri, model_name):
    api_key = AZURE_OPENAI_API_KEY
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.03,
        "max_tokens": 1024
    }
    logger.info(f"Using model: {model_name} | URI: {target_uri}")
    logger.debug(f"Prompt sent to LLM: {prompt}")
    response = requests.post(target_uri, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()['choices'][0]['message']['content']
    logger.debug(f"Response from LLM: {result}")
    return result

st.title("Raj AI by Magic Bricks")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # Add Raj's first message to the chat history
    first_message = "Hi, I'm Raj from Magic Bricks. What kind of property are you looking for today?"
    st.session_state.chat_history.append((None, first_message))
    # Also add to memory as an AI message for consistency
    if 'memory' in st.session_state:
        st.session_state.memory.chat_memory.add_ai_message(first_message)
    # Start in preference mode
    st.session_state.preference_mode = True

# Helper function to detect if a message is a preference question
PREFERENCE_KEYWORDS = [
    "looking for", "budget", "location", "type", "amenities", "bhk", "preferences", "property", "what kind of property", "what are you looking for"
]
def is_preference_message(message):
    if not message:
        return False
    msg_lower = message.lower()
    return any(keyword in msg_lower for keyword in PREFERENCE_KEYWORDS)

# Helper: Simulate preference collection (e.g., after 3+ user turns, or you can make this more advanced)
def preferences_are_collected(chat_history):
    # Count user messages that look like preferences (for demo, just count user turns)
    user_msgs = [msg for msg in chat_history if getattr(msg, 'type', None) == 'human']
    PREFERENCE_THRESHOLD = 1  # Increased from 3 to 5 to collect more preferences
    return len(user_msgs) >= PREFERENCE_THRESHOLD  # You can adjust this threshold

# Helper: Detect if user is asking for a specific property
PROPERTY_DETAIL_KEYWORDS = ["details", "show me", "about property", "property at", "tell me more", "full info", "complete info", "address is"]
def is_property_details_query(user_input):
    if not user_input:
        return False
    msg_lower = user_input.lower()
    return any(keyword in msg_lower for keyword in PROPERTY_DETAIL_KEYWORDS)

# Render chat history at the top so user messages appear instantly
st.markdown("---")
st.subheader("Chat History")
for idx, (q, a) in enumerate(st.session_state.chat_history):
    if q is not None:  # Only show user messages if they exist
        st.markdown(f"**You:** {q}")
    # For the most recent LLM response, group text and images together
    if a is not None:
        if idx == len(st.session_state.chat_history) - 1:
            property_images = st.session_state.get("last_property_images", [])
            property_names = st.session_state.get("last_property_names", []) if "last_property_names" in st.session_state else []
            property_urls = st.session_state.get("last_property_urls", []) if "last_property_urls" in st.session_state else []
            import re
            # Generalized logic for any number of numbered properties
            numbered_pattern = r'(\n|^)(\d+)\. '
            splits = [m.start() for m in re.finditer(numbered_pattern, a)]
            # Always include the start
            if splits and splits[0] != 0:
                splits = [0] + splits
            # If the number of splits matches the number of image sets, display accordingly
            if len(splits) > 1 and len(splits) - 1 == len(property_images):
                with st.container():
                    for i in range(len(property_images)):
                        part = a[splits[i]:splits[i+1]] if i+1 < len(splits) else a[splits[i]:]
                        st.markdown(f"**Raj:** {part}")
                        if property_images[i]:
                            valid_images = [img for img in property_images[i] if isinstance(img, str) and img.strip()]
                            if valid_images:
                                st.image(valid_images, width=200)
                    # Add the last part if any
                    if len(splits) > len(property_images):
                        st.markdown(f"**Raj:** {a[splits[-1]:]}")
            # Try to insert images below each property mention (fallback)
            elif property_names and property_images and any(property_names):
                answer = a
                from streamlit import markdown, image
                last_pos = 0
                output_chunks = []
                for name, images in zip(property_names, property_images):
                    if not name or not images:
                        continue
                    match = re.search(re.escape(name), answer[last_pos:], re.IGNORECASE)
                    if match:
                        start = last_pos + match.start()
                        end = last_pos + match.end()
                        output_chunks.append(answer[last_pos:end])
                        output_chunks.append(("IMAGES", images))
                        last_pos = end
                output_chunks.append(answer[last_pos:])
                with st.container():
                    for chunk in output_chunks:
                        if isinstance(chunk, tuple) and chunk[0] == "IMAGES":
                            imgs = chunk[1]
                            valid_images = [img for img in imgs if isinstance(img, str) and img.strip()]
                            if valid_images:
                                st.image(valid_images, width=200)
                        else:
                            st.markdown(f"**Raj:** {chunk}")
            else:
                with st.container():
                    st.markdown(f"**Raj:** {a}")
                    for images in property_images:
                        valid_images = [img for img in images if isinstance(img, str) and img.strip()]
                        if valid_images:
                            st.image(valid_images, width=200)
        else:
            st.markdown(f"**Raj:** {a}")

user_input = st.chat_input("Ask a question about Bangalore real estate data")
if user_input:
    # Add user message to memory
    st.session_state.memory.chat_memory.add_user_message(user_input)
    # Show user message immediately
    st.session_state.chat_history.append((user_input, None))
    st.session_state.awaiting_llm = True
    st.rerun()

# Two-stage LLM processing: if awaiting_llm is True and last chat entry has no answer
if st.session_state.get("awaiting_llm", False):
    if st.session_state.chat_history and st.session_state.chat_history[-1][1] is None:
        user_input = st.session_state.chat_history[-1][0]
        chat_history = st.session_state.memory.chat_memory.messages
        preferences_done = preferences_are_collected(chat_history)
        property_details = is_property_details_query(user_input)
        use_rag = preferences_done or property_details
        if use_rag:
            with st.spinner("Raj is typing..."):
                relevant_properties = find_relevant_context(user_input, context_data)
                context_chunks = [
                    '\n'.join([f"{k}: {v}" for k, v in prop.get('content', {}).items()])
                    for prop in relevant_properties
                ]
                urls = [prop.get('url') for prop in relevant_properties if 'url' in prop]
                property_names = []
                property_images = []  # Collect images for each property
                property_urls = []    # Collect URLs for each property
                for prop in relevant_properties:
                    content_keys = list(prop.get('content', {}).keys())
                    if content_keys:
                        prop_name = content_keys[0]
                        property_names.append(prop_name)
                        norm_name = normalize_name(prop_name)
                        url_found = url_map.get(norm_name)
                        if not url_found:
                            for k, v in url_map.items():
                                if norm_name in k or k in norm_name:
                                    url_found = v
                                    logger.warning(f"Fuzzy matched '{prop_name}' to '{k}' in url_map.")
                                    break
                    else:
                        property_names.append(None)
                        url_found = None
                    images = prop.get('content', {}).get('images', [])
                    if isinstance(images, str):
                        images = [images]
                    property_images.append(images)
                    property_urls.append(url_found)
                rag_prompt = generate_rag_prompt(user_input, context_chunks, chat_history=chat_history)
                answer = get_llm_response(rag_prompt, AZURE_GPT41mini_URI, "gpt-4.1-mini")
                logger.info(f"User Input: {user_input}\nPrompt Sent to LLM: {rag_prompt}\nLLM Output: {answer}")
                answer = re.sub(r'https?://\S+', '', answer)
                logger.debug(f"Property names for hyperlinking: {property_names}")
                logger.debug(f"url_map keys: {list(url_map.keys())}")
                logger.debug(f"Answer before hyperlinking: {answer}")
                for name, url in zip(property_names, property_urls):
                    if name and url:
                        pattern = re.compile(r'(\*\*)(' + re.escape(name) + r')(, [^\*]+)?(\*\*)', re.IGNORECASE)
                        def repl(match):
                            prefix, prop, loc, suffix = match.groups()
                            return f'{prefix}[{prop}]({url}){loc or ""}{suffix}'
                        answer, count = pattern.subn(repl, answer)
                        if count == 0:
                            logger.warning(f"Property name '{name}' not found in LLM output for hyperlinking.")
                logger.debug(f"Answer after hyperlinking: {answer}")
                st.session_state["last_property_images"] = property_images
                st.session_state["last_property_names"] = property_names
                st.session_state["last_property_urls"] = property_urls
        else:
            with st.spinner("Raj is typing..."):
                rag_prompt = generate_rag_prompt(user_input, [], chat_history=chat_history)
                answer = get_llm_response(rag_prompt, AZURE_GPT41_URI, "gpt-4.1")
                logger.info(f"User Input: {user_input}\nPrompt Sent to LLM: {rag_prompt}\nLLM Output: {answer}")
                st.session_state["last_property_images"] = []  # No images for non-RAG
        # Add assistant message to memory
        st.session_state.memory.chat_memory.add_ai_message(answer)
        # Update the last chat_history entry with the answer
        st.session_state.chat_history[-1] = (user_input, answer)
        # Update preference_mode for next turn (optional, for future use)
        st.session_state.preference_mode = is_preference_message(answer)
        st.session_state.awaiting_llm = False
        st.rerun() 

import json
import nltk
from nltk.stem import WordNetLemmatizer

# Load JSON files
intents = json.load(open("intents.json"))
seq_flow = json.load(open("conversational_flow.json"))

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

current_tag = "greeting"

def preprocess(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def intent_checker(user_input, current_tag):
    print(f"Checking for tag: 📦 {current_tag}")
    for intent in intents["intents"]:
        if intent['tag'] == current_tag:
            processed_user_input = set(preprocess(user_input))
            for pattern in intent["patterns"]:
                if set(preprocess(pattern)).issubset(processed_user_input):
                    print(f"Intent checker function returns: ↗️ {intent['responses']}")
                    return True, intent["responses"][0], intent.get("context_set")
    return False, None, None

def get_next_tag(current_tag, user_input):
    if current_tag in seq_flow:
        for next_tag, condition_tag in seq_flow[current_tag].items():
            match, _, _ = intent_checker(user_input, next_tag)
            if match:
                return condition_tag
    return None

def conversation_handler():
    global current_tag
    print("Bot: Hello! How can I assist you today?")

    while True:
        user_input = input("You: ")
        match, response, context_set = intent_checker(user_input, current_tag)

        if match:
            print(f"Bot: {response}")
            if context_set:
                current_tag = context_set
            else:
                next_tag = get_next_tag(current_tag, user_input)
                if next_tag:
                    current_tag = next_tag
                else:
                    print("Bot: I'm not sure how to proceed. Can you please rephrase or ask something else?")
        else:
            print("Bot: I'm sorry, I didn't understand that. Can you please rephrase or ask something else?")

        if current_tag == "end_call":
            print("Bot: Thank you for your time. Have a great day!")
            break

if __name__ == "__main__":
    conversation_handler()
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import webbrowser
import datetime

class chatbotgui:
    def __init__(self, master):
        self.master=master
        self.setup_gui()
        self.load_chatbot_data()
        self.conversation_history=[]

    def setup_gui(self):
        self.master.title("Gym-bot")
        self.master.geometry("500x600")
        self.master.configure(bg="#f0f0f0")

        style=ttk.Style()
        style.theme_use("clam")

        main_frame=ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
#chat history box
        self.chat_history= scrolledtext.ScrolledText(
        main_frame, wrap=tk.WORD, width=60, height=25, font=("Arial", 10)
    )
        self.chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_history.config(state=tk.DISABLED)

#input and button frame

        input_frame=ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        self.user_input=ttk.Entry(input_frame, width=50, font=("Arial", 10))
        self.user_input.pack(side=tk.LEFT, padx=(0,5), expand=True, fill=tk.X)
        self.user_input.bind("<Return>",lambda event:self.send_message())

        self.send_button=ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

#button frame
        button_frame=ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.clear_button=ttk.Button(
        button_frame, text="Clear Chat", command=self.clear_chat
    )
        self.clear_button.pack(side=tk.LEFT, padx=(0,5))

        self.save_button=ttk.Button(button_frame, text="Save Chat", command=self.save_chat)
        self.save_button.pack(side=tk.LEFT)

        self.help_button=ttk.Button(button_frame, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.RIGHT)

#loading chatbot data
    def load_chatbot_data(self):
        self.lemmatizer= WordNetLemmatizer()
        self.intents=json.loads(open(r"C:/Users/91754/Desktop/Chatbot/intents.json").read())
        self.words=pickle.load(open("words.pkl", "rb"))
        self.classes=pickle.load(open("classes.pkl", "rb"))
        self.model=load_model("chatbot.h5")

#sending and recieving messages
    def send_message(self):
        user_message=self.user_input.get()
        self.user_input.delete(0, tk.END)
        if user_message:
            self.update_chat_history(f"You:{user_message}")
            bot_response=self.get_bot_response(user_message)
            self.update_chat_history(f"Bot:{bot_response}")
            self.conversation_history.append((user_message, bot_response))

#update chat hsitory
    def update_chat_history(self, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

#bot response
    def get_bot_response(self, user_message):
        if user_message.lower() in ["exit", "quit", "bye"]:
            return "Goodbye!"
        elif user_message.lower().startswith("search "):
            query = user_message[7:]
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"I have opened the web search for '{query}"
        elif user_message.lower()=="time":
            return(
                f"the current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
            )
        else:
            ints=self.predict_class(user_message)
            return self.get_response(ints)
        
#cleaning sentence
    def clean_sentence(self, sentence):
        return[
            self.lemmatizer.lemmatize(word.lower())
            for word in nltk.word_tokenize(sentence)

        ]
    def bag_o_words(self,sentence):
        sentence_words=self.clean_sentence(sentence)
        bag=[1 if word in sentence_words else 0 for word in self.words]
        return np.array(bag)

    
#predicting class
    def predict_class(self, sentence):
        bow=self.bag_o_words(sentence)
        res= self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD=0.25
        results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {"intent": self.classes[r[0]], "probability": str(r[1])} for r in results
        ]
    def get_response(self, intents_list):
        if not intents_list:
            return"not sure please rewrite"
        tag = intents_list[0]["intent"]
        for intent in self.intents["intents"]:
            if intent["tag"]==tag:
                return random.choice(intent["responses"])
        return "Im sorry, ask again please"
    
#clearind + saving
    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)
        self.conversation_history.clear()

    def save_chat(self):
        filename=(f"chat_history_{datetime,datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filename, "w") as f:
            for user_msg, bot_msg in self.conversation_history:
                f.write(f"You:{user_msg}\n")
                f.write(f"Bot:{bot_msg}\n\n")
        messagebox.showinfo("chat saved", f"chat saved to {filename}")

#show help
    def show_help(self):
        help_text="""Welcome to the bot!
        not help so far"""
        messagebox.showinfo("ChatbotHelp", help_text)

#run the app
if __name__=="__main__":
    root=tk.Tk()
    chatbotgui(root)
    root.mainloop()
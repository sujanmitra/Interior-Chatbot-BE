from chatterbot import ChatBot

from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot=ChatBot('bro')

# TODO: Train with custom dataset
trainer = ChatterBotCorpusTrainer(chatbot)

trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english.conversations" )
 
while True:
       user_input = input("You: ")
       if user_input.lower() == 'bye':
           print("Byeeeeee")
           break
       response = chatbot.get_response(user_input)
       print("Bro:", response)
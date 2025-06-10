import tensorflow as tf
from chatbot_train import response
from detect import main

print("Note: Enter 'quit' to break the loop.")   
while True:                                                
    query = input('You: ')
    if query.lower() == 'quit':
        break
    bot_response, typ = response(query)
    if typ == 'ActionQuery':
        main('../testdata/sofa2.webp', query)
    print('Geek: {} -- TYPE: {}'.format(bot_response, typ))
    print()
from chatbot_train import response
from detect import process_img

def start_chat(img_path, prompt):
    print("Note: Enter 'quit' to break the loop.")   
    while True:
        # TODO: Add escape sequence
        bot_response, typ = response(prompt)
        if typ == 'ActionQuery':
            process_img(img_path, prompt)
        return bot_response
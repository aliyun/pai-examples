import copy
import re
import gradio as gr
import base64
from argparse import ArgumentParser
def _get_args():
    parser = ArgumentParser()

    parser.add_argument('--eas_endpoint', type=str, required=True)
    parser.add_argument('--eas_token', type=str, required=True)
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    return args

def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def encode_base64_content_from_file(file_path):
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return encoded_string


def transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image_url' in item:
                new_item = {'type': 'image_url', 'image_url': {"url":item['image_url']}}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video_url' in item:
                new_item = {'type': 'video_url', 'video_url': {"url": item['video_url']}}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _launch_ui(model, client, args):

    def call_local_model(messages, max_tokens, temperature, timeout, use_stream):
        try:
            messages = transform_messages(messages)
            print('Messages:', messages)
            print(f"Arguments: max_tokens={max_tokens}, temperature={temperature}, timeout={timeout}, stream={use_stream}")
            gen = client.chat.completions.create(
                messages = messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                stream=use_stream
            )
            if use_stream:
                generated_text = ''
                for chunk in gen:
                    generated_text += chunk.choices[0].delta.content
                    yield generated_text
            else:
                generated_text = gen.choices[0].message.content  
                yield generated_text
        except Exception as e:
            print(e)
            raise gr.Error(e.message)

    def create_predict_fn():

        def predict(_chatbot, task_history, max_tokens, temperature, timeout, use_stream):
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for cls, q, a in history_cp:
                if cls == 'image':
                    content.append({'image_url': f"data:image/jpeg;base64,{q}"})
                elif cls == 'video':
                    content.append({'video_url': f"data:video/mp4;base64,{q}"})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(messages, max_tokens, temperature, timeout, use_stream):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = ('text', query, full_response)
            print('VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict

    def create_regenerate_fn():

        def regenerate(_chatbot, task_history, max_tokens, temperature, timeout, use_stream):
            print("Regenerating...")
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            _chatbot_gen = predict(_chatbot, task_history, max_tokens, temperature, timeout, use_stream)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [('text', task_text, None)]
        return history, task_history

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        base64_encoded_content = encode_base64_content_from_file(file.name)
        history = history + [((file.name,), None)]
        if is_video_file(file.name):
            task_history = task_history + [('video', base64_encoded_content, None)]
        else:
            task_history = task_history + [('image', base64_encoded_content, None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(task_history):
        print('Clear history.')
        task_history.clear()
        return []

    with gr.Blocks() as demo:

        gr.Markdown(f"""<center><font size=8>MLLM-WebUI</center>""")
        chatbot = gr.Chatbot(elem_classes='control-height', height=480)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])
        
        with gr.Row():
            max_tokens = gr.Slider(minimum=10, maximum=10240, step=10, label="max_tokens", value=1024)
            temperature = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, label="temperature", value=0.0)
            timeout = gr.Slider(minimum=0, maximum=600, step=10, label="timeout(seconds)", value=120)
            use_stream_chat = gr.Checkbox(label="use_stream_chat", value=True)
        
        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit (发送)')
            regen_btn = gr.Button('🤔️ Regenerate (重试)')
            empty_bin = gr.Button('🧹 Clear History (清除历史)')  
            
        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history]).then(predict, 
                         [chatbot, task_history, max_tokens, temperature, timeout, use_stream_chat], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history, max_tokens, temperature, timeout, use_stream_chat], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

def main():
    from openai import OpenAI
    args = _get_args()
    openai_api_key = args.eas_token
    if not args.eas_endpoint.endswith("/"):
        args.eas_endpoint += "/"
    openai_api_base = f"{args.eas_endpoint}v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base) 
    models = client.models.list()
    model = models.data[0].id
    _launch_ui(model, client, args)
 
if __name__ == '__main__':
    main()
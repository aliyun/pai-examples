import gradio as gr
from argparse import ArgumentParser
import json
import requests


def _get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--eas_endpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--eas_token",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
    )

    args = parser.parse_args()
    return args


css = """
.checkbox {
    max-width: 2.5em;
    min-width: 2.5em !important;
    display:flex;
    align-items:center;
}
"""


def _launch_ui(model_name, client, args):
    def _transform_messages(history, max_rounds, apply_max_rounds, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if not apply_max_rounds:
            begin_index = 0
        else:
            begin_index = max(0, len(history) - max_rounds)

        for i in range(begin_index, len(history)):
            query, response = history[i]
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": response})
        messages.pop()  # pop the None assistant response
        return messages

    def predict(
        _chatbot,
        max_tokens,
        top_k,
        apply_top_k,
        top_p,
        apply_top_p,
        temperature,
        apply_temperature,
        use_stream,
        max_rounds,
        apply_max_rounds,
        system_prompt,
    ):
        chat_query = _chatbot[-1][0]
        if len(chat_query) == 0:
            _chatbot.pop()
            return _chatbot
        messages = _transform_messages(
            _chatbot, max_rounds, apply_max_rounds, system_prompt
        )
        print(f"Messages: {json.dumps(messages)}")
        request = {"messages": messages, "stream": use_stream, "max_tokens": max_tokens}
        if apply_temperature:
            request["temperature"] = temperature
        if apply_top_k:
            request["top_k"] = top_k
        if apply_top_p:
            request["top_p"] = top_p
        response = requests.post(
            url=f"{args.eas_endpoint}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": args.eas_token,
            },
            json=request,
            stream=use_stream,
        )
        if use_stream:
            generated_text = ""
            for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False):
                msg = chunk.decode("utf-8")
                if msg.startswith("data"):
                    info = msg[6:]
                    if info == "[DONE]":
                        break
                    else:
                        resp = json.loads(info)
                        generated_text += resp["choices"][0]["delta"]["content"]
                        _chatbot[-1] = (chat_query, generated_text)
                        yield _chatbot
        else:
            resp = json.loads(response.text)
            generated_text = resp["choices"][0]["message"]["content"]
            _chatbot[-1] = (chat_query, generated_text)
            yield _chatbot

    def add_text(history, text):
        history = history if history is not None else []
        history.append([text, None])  # [user_query, bot_response]
        return history, None

    def clear_history(history):
        if history:
            history.clear()
        return []

    with gr.Blocks(analytics_enabled=False, css=css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""<h2><center>ChatLLM-WebUI</center></h2>""")
                if model_name:
                    gr.Markdown(f"""<h3><center>{model_name}</center></h3>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        model_argument = gr.Accordion("Model Arguments")
                        with model_argument:
                            with gr.Row():
                                max_tokens = gr.Slider(
                                    minimum=10,
                                    maximum=10240,
                                    step=10,
                                    label="max_tokens",
                                    value=512,
                                )
                            with gr.Row():
                                apply_top_k = gr.Checkbox(
                                    label="", value=True, elem_classes="checkbox"
                                )
                                top_k = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    step=1,
                                    label="top_k",
                                    value=10,
                                )
                            with gr.Row():
                                apply_top_p = gr.Checkbox(
                                    label="", value=False, elem_classes="checkbox"
                                )
                                top_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label="top_p",
                                    value=0,
                                )
                            with gr.Row():
                                apply_temperature = gr.Checkbox(
                                    label="", value=True, elem_classes="checkbox"
                                )
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.01,
                                    label="temperature",
                                    value=0.0,
                                )

                            with gr.Row():
                                use_stream_chat = gr.Checkbox(
                                    label="use_stream_chat", value=True
                                )

                        with gr.Row():
                            max_rounds = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                label="max_rounds",
                                value=10,
                            )
                            apply_max_rounds = gr.Checkbox(
                                label="", value=True, elem_classes="checkbox"
                            )

                        with gr.Row():
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                lines=4,
                                value="You are a helpful assistant.",
                            )
                            clear_prompt_btn = gr.Button("Clear Prompt")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False, height=560)
                with gr.Row():
                    query = gr.Textbox(label="Input", lines=3)

                with gr.Row():
                    submit_btn = gr.Button("submit", elem_id="c_generate")
                    clear_history_btn = gr.Button("clear history")

        submit_btn.click(add_text, [chatbot, query], [chatbot, query]).then(
            predict,
            [
                chatbot,
                max_tokens,
                top_k,
                apply_top_k,
                top_p,
                apply_top_p,
                temperature,
                apply_temperature,
                use_stream_chat,
                max_rounds,
                apply_max_rounds,
                system_prompt,
            ],
            [chatbot],
            show_progress=True,
        )
        clear_history_btn.click(clear_history, [chatbot], [chatbot], show_progress=True)
        clear_prompt_btn.click(lambda: "", None, [system_prompt])

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    _launch_ui(model_name=None, client=None, args=args)


if __name__ == "__main__":
    main()

import gradio as gr
from argparse import ArgumentParser
import json
from openai import OpenAI
from openai import NOT_GIVEN


def _get_args():
    parser = ArgumentParser()

    parser.add_argument("--eas_endpoint", type=str, required=True)
    parser.add_argument("--eas_token", type=str, required=True)
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
    def _post_process(text):
        return text.replace("<think>", "&lt;think&gt;").replace(
            "</think>", "&lt;/think&gt;"
        )

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
        max_completion_tokens,
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
        print(f"Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        gen = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p if apply_top_p else NOT_GIVEN,
            temperature=temperature if apply_temperature else NOT_GIVEN,
            stream=use_stream,
        )
        print("Response:", end="")
        if use_stream:
            generated_text = ""
            for chunk in gen:
                generated_text += _post_process(chunk.choices[0].delta.content)
                print(chunk.choices[0].delta.content, end="")
                _chatbot[-1] = (chat_query, generated_text)
                yield _chatbot
        else:
            generated_text = _post_process(gen.choices[0].message.content)
            print(gen.choices[0].message.content, end="")
            _chatbot[-1] = (chat_query, generated_text)
            yield _chatbot
        print()

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
                gr.Markdown(f"""<h3><center>{model_name}</center></h3>""")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        model_argument = gr.Accordion("Model Arguments")
                        with model_argument:
                            with gr.Row():
                                max_completion_tokens = gr.Slider(
                                    minimum=10,
                                    maximum=10240,
                                    step=10,
                                    label="max_completion_tokens",
                                    value=512,
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
                                    value=0.7,
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
                max_completion_tokens,
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
    openai_api_key = args.eas_token
    if not args.eas_endpoint.endswith("/"):
        args.eas_endpoint += "/"
    openai_api_base = f"{args.eas_endpoint}v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    _launch_ui(model, client, args)


if __name__ == "__main__":
    main()

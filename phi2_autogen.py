from autogen import AssistantAgent, UserProxyAgent

llm_config = {
    "config_list": [
        {
            "model": "phi",
            "api_key": "EMPTY",
            "base_url": "http://localhost:11434/v1",
            "price": [0, 0]
        }
    ]
}

assistant = AssistantAgent(name="Assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(name="User", human_input_mode="ALWAYS")  # Fixed to 'ALWAYS'

user_proxy.initiate_chat(
    assistant,
    message="Hello! Let's start an interactive coding session."
)

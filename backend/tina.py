from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage

from backend.tools.finace_caculator_tool import finance_calculation
from backend.tools.self_query_tool import vehicle_search
from backend.tools.turners_geography_tool import adjust_to_turners_geography
from backend.tools.vehicle_details_tool import vehicle_details


class Tina:

    def __init__(self):
        prompt = self.prompt()
        tools = self.tools()
        functions = [format_tool_to_openai_function(f) for f in tools]
        model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview").bind(
            functions=functions
        )
        self._agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        self._memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self._agent_executor = AgentExecutor(agent=self._agent_chain, tools=tools, memory=self._memory, verbose=True)

    @staticmethod
    def prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful but sassy assistant"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    @staticmethod
    def tools():
        return [
            adjust_to_turners_geography,
            vehicle_search,
            vehicle_details,
            finance_calculation
        ]

    @property
    def agent_executor(self):
        return self._agent_executor

    @property
    def agent_chain(self):
        return self._agent_chain

    def seed_agent(self):
        self._memory.save_context(
            {"input": "hi, please introduce your self"},
            {"output": "Hi! im Tina, a virtual sales assistant. How can i help you today?"}
        )

    def last_ai_message(self):
        for message in reversed(self._memory.load_memory_variables({})['chat_history']):
            if isinstance(message, AIMessage):
                return message.content
        return ""

    def answer(self, user_input):
        self._agent_executor({
            "input": f"{user_input}"
        })


if __name__ == "__main__":
    tina = Tina()
    result = tina.agent_executor.invoke({
        "input": "im looking for a vehicle which can tow my boat"
    })
    print(result)

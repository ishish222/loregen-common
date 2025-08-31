# Pseudo-random Story Generator

import random
from langchain_core.runnables import RunnableLambda, Runnable
from typing import Sequence, TypeVar, Generic, Type, cast, get_args
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import ChatPromptValue
from pydantic import BaseModel, RootModel
from langchain_core.language_models import LanguageModelInput


# We need to modify the inference pipeline to create a spectrum of stories.
# input: a regular prompt from the user that would've been forwarded to model
# output: a prompt with added conditions on the output
# So the general conditions will be in the SYSTEM message, we need to focus on the reasoning message

# examples of reasoning prompts in history generation:
# f"We currently have {len(state.history_world)}/{state.number_of_epochs} epochs in the world. Please add another epoch to the world unless we're done."
# f"We currently have {len(state.history_country)}/{len(state.history_world)} epochs in the country's history. Please add another epoch to the world unless we're done."
# f"We currently have {len(state.history_city)}/{len(state.history_country)} epochs in the history of the city. Please add another epoch to the history unless we're done."
# f"We currently have {len(state.history_family)}/{state.number_of_generations} generations in the family. Please add another generation to the family unless we're done."
# f"We currently have {len(state.history_character)}/{state.number_of_chapters} chapters in the story. Please add another chapter to the story unless we're done."


T = TypeVar("T", bound=BaseModel)


def _pr_pick(
    items: Sequence[T]
) -> T:
    if not items:
        raise ValueError("Cannot pick a random item from an empty sequence")
    return random.choice(items)


pr_pick: RunnableLambda[Sequence[T], T] = RunnableLambda(_pr_pick)


class ListOf(RootModel[list[T]], Generic[T]):
    pass


class RandomPickStructured(Generic[T], Runnable[LanguageModelInput, T]):
    '''
    We need this class in order to bind the generic types between the chat model with
    the structured output and the pr_pick runnable.
    '''

    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        schema: Type[T] | None = None,
    ):
        if schema is None:
            try:
                schema = cast(Type[T], get_args(self.__orig_class__)[0])
            except Exception as e:
                raise TypeError(
                    "Schema not provided and could not infer T. "
                    "Either pass `schema=...` or instantiate as RandomPickStructured[YourSchema](...)."
                ) from e

        structured_cls = ListOf[schema]  # type: ignore[valid-type]
        self._model = chat_model.with_structured_output(structured_cls)
        self._model = cast(Runnable[LanguageModelInput, ListOf[T]], self._model)

        # Create chain with properly bound types
        self._chain: Runnable[LanguageModelInput, T] = self._model | (lambda x: x.root) | pr_pick

    def invoke(self, messages: LanguageModelInput, config=None) -> T:
        return self._chain.invoke(messages, config)

    async def ainvoke(self, messages: LanguageModelInput, config=None) -> T:
        return await self._chain.ainvoke(messages, config)


def _preprocess_chat_prompt(
    chat_prompt: ChatPromptValue,
    *,
    number_of_responses: int = 10,
) -> ChatPromptValue:

    if not isinstance(chat_prompt.messages[-1], HumanMessage):
        raise ValueError("The last message in the template must be a human message")

    additional_message = HumanMessage(content=f'''
        Please generate a list of {number_of_responses} that cover the full spectrum from fortunate to unfortunate.
        '''
    )

    new_reasoning_message = chat_prompt.messages[-1] + additional_message

    return ChatPromptValue(messages=[*chat_prompt.messages[:-1], new_reasoning_message])


pr_preprocess: RunnableLambda[ChatPromptValue, ChatPromptValue] = RunnableLambda(_preprocess_chat_prompt)

# Usage:
# picker = RandomPickStructured[ReasoningInfancyResponseSchema](
#    chat_model=ChatOpenAI(model="gpt-4o-mini", temperature=0)
# )
# mychain = prompt | pr_preprocess | picker
# mychain.invoke({**prompt_args, number_of_responses=10})
#
# or:
# mychain = pr_process.bind(number_of_responses=10) | picker
# mychain.invoke({**prompt_args})

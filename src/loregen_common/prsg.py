# Pseudo-random Story Generator

import random
from langchain_core.runnables import RunnableLambda, Runnable
from typing import Sequence, TypeVar, Generic, Type, cast, get_args
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptValue
from pydantic import BaseModel, ConfigDict, create_model, Field
from langchain_core.language_models import LanguageModelInput

random.seed()
T = TypeVar("T", bound=BaseModel)


def make_candidates_model(
    schema: Type[T],
    *,
    name: str = "Candidates",
) -> Type[BaseModel]:
    # name becomes the Python class name (and typically the schema title source)
    cls = create_model(
        name,
        candidates=(list[schema], Field(..., description="List of candidates")),
        __base__=BaseModel,
        __module__="loregen_schemas",  # avoids "<locals>" in some toolchains
    )
    cls.model_config = ConfigDict(title=name)  # short/safe title
    return cls


def _pr_pick(
    items: Sequence[T]
) -> T:
    if not items:
        raise ValueError("Cannot pick a random item from an empty sequence")
    selected = random.choice(items)
    return selected


pr_pick: RunnableLambda[Sequence[T], T] = RunnableLambda(_pr_pick)


# class CandidatesWrapper(BaseModel, Generic[T]):
#     model_config = ConfigDict(title="Candidates")  # short title
#     candidates: list[T]


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

        # structured_cls = CandidatesWrapper[schema]  # type: ignore[valid-type]

        CandidatesModel = make_candidates_model(schema, name="Candidates")

        self._model = chat_model.with_structured_output(
            CandidatesModel,
            method="json_schema",
            strict=True,
        )
        self._chain: Runnable[LanguageModelInput, T] = self._model | (lambda x: x.candidates) | pr_pick

    def invoke(self, messages: LanguageModelInput, config=None) -> T:
        return self._chain.invoke(messages, config)

    async def ainvoke(self, messages: LanguageModelInput, config=None) -> T:
        return await self._chain.ainvoke(messages, config)


def _preprocess_chat_prompt(
    chat_prompt: ChatPromptValue,
    *,
    number_of_responses: int = 6,
) -> ChatPromptValue:
    msgs = list(chat_prompt.messages)
    if not msgs:
        raise ValueError("Prompt has no messages")

    last = msgs[-1]
    if not isinstance(last, HumanMessage):
        raise ValueError("The last message in the template must be a HumanMessage")

    extra = (
        f"""

        Please generate a list of {number_of_responses} candidate answers spanning
        from very fortunate to very unfortunate. Make sure you cover the full spectrum between
        fortunate and unfortunate.

        For example, if you are asked for 10 candidate answers, you might assign something like:

        - Candidates [0], [1], [2] must describe **severe adversity** (war, neglect, illness, violence, abandonment, poverty, etc.).
        - Candidates [3], [4], [5] should describe **mixed or ambiguous** circumstances (both supportive and adverse elements).
        - Candidates [6], [7], [8], [9] should describe **positive and fortunate** circumstances (love, prosperity, creativity, trust).

        Make each candidate distinct, not variations on the same theme.
        """
    )

    msgs[-1] = HumanMessage(content=f"{last.content}\n\n{extra}")
    return ChatPromptValue(messages=msgs)


pr_preprocess: RunnableLambda[ChatPromptValue, ChatPromptValue] = RunnableLambda(_preprocess_chat_prompt)

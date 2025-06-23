import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

# Skip tests if core dependencies are missing
np = pytest.importorskip("numpy")
pytest.importorskip("requests")
pytest.importorskip("ray")
pytest.importorskip("omegaconf")

from verl.tools.search_tool import SearchTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from verl.tools.utils.document_indexer import DocumentIndexer


@pytest.fixture
def tool_schema():
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="search",
            description="Search",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "query_list": OpenAIFunctionPropertySchema(
                        type="array",
                        description="queries",
                        items={"type": "string"},
                    )
                },
                required=["query_list"],
            ),
        ),
    )


@pytest.fixture
def search_tool(tool_schema):
    config = {"retrieval_service_url": "local"}
    return SearchTool(config, tool_schema)


@pytest.mark.asyncio
async def test_create_stores_indexer(search_tool):
    instance_id = await search_tool.create(support="Paris is beautiful.")
    assert isinstance(search_tool._instance_dict[instance_id]["indexer"], DocumentIndexer)


@pytest.mark.asyncio
async def test_execute_uses_local_indexer(search_tool):
    support = "Paris is the capital city of France. It is famous for the Eiffel Tower."
    instance_id = await search_tool.create(support=support)

    async def fake_remote(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    search_tool.execution_pool.execute.remote = AsyncMock(side_effect=fake_remote)

    indexer = search_tool._instance_dict[instance_id]["indexer"]
    with patch.object(indexer, "batch_search", wraps=indexer.batch_search) as mock_batch, patch(
        "verl.tools.search_tool.perform_single_search_batch"
    ) as mock_api:
        result_text, reward, metrics = await search_tool.execute(
            instance_id, {"query_list": ["Eiffel Tower"]}
        )

        assert mock_batch.called
        mock_api.assert_not_called()

    result = json.loads(result_text)["result"]
    assert "Eiffel" in result
    assert metrics["status"] == "success"

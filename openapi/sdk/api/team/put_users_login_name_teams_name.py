from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.team_spec import TeamSpec
from ...models.team_update_request import TeamUpdateRequest
from ...types import Response


def _get_kwargs(
    login_name: str,
    name: str,
    *,
    json_body: TeamUpdateRequest,
) -> Dict[str, Any]:
    pass

    json_json_body = json_body.to_dict()

    return {
        "method": "put",
        "url": "/users/{login_name}/teams/{name}".format(
            login_name=login_name,
            name=name,
        ),
        "json": json_json_body,
    }


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[TeamSpec]:
    if response.status_code == HTTPStatus.OK:
        response_200 = TeamSpec.from_dict(response.json())

        return response_200
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[TeamSpec]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    login_name: str,
    name: str,
    *,
    client: AuthenticatedClient,
    json_body: TeamUpdateRequest,
) -> Response[TeamSpec]:
    """Update the team.

     Update the team.

    Args:
        login_name (str):
        name (str):
        json_body (TeamUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[TeamSpec]
    """

    kwargs = _get_kwargs(
        login_name=login_name,
        name=name,
        json_body=json_body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    login_name: str,
    name: str,
    *,
    client: AuthenticatedClient,
    json_body: TeamUpdateRequest,
) -> Optional[TeamSpec]:
    """Update the team.

     Update the team.

    Args:
        login_name (str):
        name (str):
        json_body (TeamUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        TeamSpec
    """

    return sync_detailed(
        login_name=login_name,
        name=name,
        client=client,
        json_body=json_body,
    ).parsed


async def asyncio_detailed(
    login_name: str,
    name: str,
    *,
    client: AuthenticatedClient,
    json_body: TeamUpdateRequest,
) -> Response[TeamSpec]:
    """Update the team.

     Update the team.

    Args:
        login_name (str):
        name (str):
        json_body (TeamUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[TeamSpec]
    """

    kwargs = _get_kwargs(
        login_name=login_name,
        name=name,
        json_body=json_body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    login_name: str,
    name: str,
    *,
    client: AuthenticatedClient,
    json_body: TeamUpdateRequest,
) -> Optional[TeamSpec]:
    """Update the team.

     Update the team.

    Args:
        login_name (str):
        name (str):
        json_body (TeamUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        TeamSpec
    """

    return (
        await asyncio_detailed(
            login_name=login_name,
            name=name,
            client=client,
            json_body=json_body,
        )
    ).parsed

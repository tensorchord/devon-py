from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.deployment_instance import DeploymentInstance


T = TypeVar("T", bound="InstanceListResponse")


@_attrs_define
class InstanceListResponse:
    """
    Attributes:
        instances (Union[Unset, List['DeploymentInstance']]):
    """

    instances: Union[Unset, List["DeploymentInstance"]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        instances: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.instances, Unset):
            instances = []
            for instances_item_data in self.instances:
                instances_item = instances_item_data.to_dict()

                instances.append(instances_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instances is not UNSET:
            field_dict["instances"] = instances

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.deployment_instance import DeploymentInstance

        d = src_dict.copy()
        instances = []
        _instances = d.pop("instances", UNSET)
        for instances_item_data in _instances or []:
            instances_item = DeploymentInstance.from_dict(instances_item_data)

            instances.append(instances_item)

        instance_list_response = cls(
            instances=instances,
        )

        instance_list_response.additional_properties = d
        return instance_list_response

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

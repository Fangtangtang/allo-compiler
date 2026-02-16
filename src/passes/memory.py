# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from allo.memory import Layout, Memory, Offset4D


class DTensor:
    """
    Distributed tensor.
    """

    def __init__(
        self,
        mapping,
        shape,
        dtype,
        spec_list,
        tile_shape: list,
        is_input: bool,
        id: int,
    ):
        self.mapping = mapping  # mesh dims
        self.shape = shape  # tensor shape
        self.dtype = dtype
        self.global_id: int = id
        self.is_input: bool = is_input
        self.tile_shape = tile_shape

        # Handle spec
        self.layout: Layout = None
        self.memory: Memory = None
        for spec in spec_list:
            if isinstance(spec, Layout):
                self.layout = spec
            elif isinstance(spec, Memory):
                self.memory = spec
            else:
                raise RuntimeError(f"Fail to resolve spec {spec}")

        if self.layout is not None:
            assert mapping is not None
            # tensor tile ID -> PE tile IDs
            self.global_placement: dict[
                tuple[int | str, ...], list[tuple[int, ...]]
            ] = self.layout.get_placement(mapping)
            self.set_access_pattern()

    def set_access_pattern(self):
        """
        Specify how to access the dtensor (local tensor) from the global tensor
            (tensor has at most 4 dimensions: DMA support 4-dimension address generation)
        Set offset map for each tensor tile.

        Returns:
            - device_dims (list): Indexes of tensor dimensions sharded across devices.
            - size (list): 4D tensor dimensions used for access.
            - stride (list): Stride along each dimension in the global tensor.
        """
        # tensor tile ID -> address offset
        self.offset_map: dict[tuple[int | str, ...], Offset4D] = {}
        partition_str = "".join(
            [
                "S" if isinstance(p, Layout.Shard) else "R"
                for p in self.layout.partitions
            ]
        )
        partition_dim = [
            p.axis if isinstance(p, Layout.Shard) else None
            for p in self.layout.partitions
        ]
        if len(self.shape) == 1:
            if partition_str == "S":
                dim = partition_dim[0]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(0, 0, i, 0)
                shard_size = self.shape[0] // self.mapping[dim]
                device_dims = [2]  # partition idx = 2
                size = [1, 1, self.mapping[dim], shard_size]
                stride = [0, 0, shard_size, 1]
            elif partition_str == "R":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
                device_dims = []  # no partition
                size = [1, 1, 1, self.shape[0]]
                stride = [0, 0, 0, 1]
            else:
                raise ValueError("Unsupported access pattern for 1D tensor.")
        elif len(self.shape) == 2:
            tensor_m, tensor_n = self.shape  # [tensor_m x tensor_n]
            if partition_str == "SS":
                device_a, device_b = (
                    self.mapping[partition_dim[0]],
                    self.mapping[partition_dim[1]],
                )
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                device_dims = [0, 1]
                size = [device_a, device_b, tensor_m // device_a, tensor_n // device_b]
                stride = [
                    (tensor_m // device_a) * tensor_n,
                    tensor_n // device_b,
                    tensor_n,
                    1,
                ]
            elif partition_str == "SR":
                device_a = self.mapping[partition_dim[0]]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_a, i % device_a, 0, 0)
                # First dim sharded across all devices, second replicated
                device_dims = [1]
                size = [1, device_a, tensor_m // device_a, tensor_n]
                stride = [0, (tensor_m // device_a) * tensor_n, tensor_n, 1]
            elif partition_str == "RS":
                device_b = self.mapping[partition_dim[1]]
                for i, key in enumerate(sorted(list(self.global_placement.keys()))):
                    self.offset_map[key] = Offset4D(i // device_b, i % device_b, 0, 0)
                # First dim replicated, second sharded across second dim of mesh
                device_dims = [1]
                size = [1, device_b, tensor_m, tensor_n // device_b]
                stride = [0, tensor_n // device_b, tensor_n, 1]
            elif partition_str == "RR":
                for key in self.global_placement.keys():
                    self.offset_map[key] = Offset4D(0, 0, 0, 0)
                # Both dimensions replicated
                device_dims = []
                size = [1, 1, tensor_m, tensor_n]
                stride = [0, 0, tensor_n, 1]
            else:
                raise ValueError("Unsupported access pattern for 2D tensor.")
        else:
            raise ValueError("Unsupported access pattern.")
        self.shared_dims, self.size, self.stride = device_dims, size, stride

    def PE_tile_id_to_tensor_tile_id(
        self, pe_tile_id: tuple[int, ...]
    ) -> tuple[int | str, ...]:
        for tensor_tile_id, pe_tile_ids in self.global_placement.items():
            if pe_tile_id in pe_tile_ids:
                return tensor_tile_id
        raise ValueError(
            f"PE tile ID {pe_tile_id} not found in {self.global_placement}"
        )

    def __str__(self):
        parts = [
            f"shape={self.shape}",
            f"dtype={self.dtype}",
        ]
        if self.layout is not None:
            parts.append(f"layout={self.layout}")
        if self.memory is not None:
            parts.append(f"memory={self.memory}")
        parts.extend(
            [
                f"mapping={self.mapping}",
                f"tile_shape={self.tile_shape}",
            ]
        )
        return f"DTensor({', '.join(parts)})"

# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from cirq.study import resolver

"""Defines ComputeDisplaysResult."""


class ComputeDisplaysResult:
    """Results of computing the values of displays in a circuit.

    Attributes:
        params: A ParamResolver of settings used for this result.
        display_values: A dictionary from display key to display value.
    """

    def __init__(self,
                 params: resolver.ParamResolver,
                 display_values: Dict) -> None:
        self.params = params
        self.display_values = display_values

    def __repr__(self):
        return ('cirq.ComputeDisplaysResult('
                'params={!r}, '
                'display_values={!r})').format(self.params,
                                               self.display_values)
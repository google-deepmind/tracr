# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RASP program objects.

Every object in the RASP language is a function.

The most important type is S-Op, which is a function List[Value] -> List[Value].

An S-Op represents a state inside the residual stream of the transformer.
Therefore, any RASP program that represents a transformer computation must
define a final S-Op that represents the state of the residual stream at the
end of the computation. In particular, given an S-Op `x`,
`x([1, 2, 3])` represents something like the state of the residual stream
at location `x` when the transformer is fed [1, 2, 3] as input.

A secondary (but still important) type is Selector, which is a function
List[Value] -> List[List[bool]]. Given a Selector `sel`, sel([1, 2, 3])
represents something like an attention matrix in the transformer.

For a full reference on RASP, see https://arxiv.org/abs/2106.06981.
"""

import abc
import collections.abc
import copy
import enum
import functools
import itertools
from typing import (Any, Callable, Dict, Generic, List, Mapping, Optional,
                    Sequence, TypeVar, Union)

from absl import logging
import numpy as np
from typing_extensions import Protocol

SelectorValue = List[List[bool]]
NumericValue = Union[int, float]
Value = Union[None, int, float, str, bool]
VT = TypeVar("VT", bound=Value)
RASPExprT = TypeVar("RASPExprT", bound="RASPExpr")
SOpT = TypeVar("SOpT", bound="SOp")
T = TypeVar("T")

_NAME_KEY = "name"
_ENCODING_KEY = "encoding"

# These are run on every expression when it's initialised.
# Add your own annotators to this dict to add custom default annotations.
#
# For example, DEFAULT_ANNOTATORS['foo'] will provide the default value for
# expr.annotations['foo]. The annotator will get called lazily the first time
# that key is accessed.
#
# See the `default_name` annotator for a full example.
DEFAULT_ANNOTATORS: Dict[str, "Annotator"] = {}


class Annotator(Protocol):

  def __call__(self, expr: "RASPExpr") -> Any:
    """What annotation to add to `expr`."""


class _Annotations(collections.abc.Mapping):
  """Holds the expression's annotations.

  It's immutable to the user, but will attempt to generate default values
  lazily when missing keys are requested.
  """

  def __init__(self, expr, **kwargs: Any):
    self._expr = expr
    self._inner_dict: Dict[str, Any] = {**kwargs}

  def __getitem__(self, key: str) -> Any:
    if key not in self._inner_dict:
      if key not in DEFAULT_ANNOTATORS:
        raise KeyError(
            f"No annotation exists for key '{key}'. Available keys:"
            f" {set(self.keys()) | set(DEFAULT_ANNOTATORS.keys())}"
        )
      self._inner_dict[key] = DEFAULT_ANNOTATORS[key](self._expr)

    return self._inner_dict[key]

  def __iter__(self):
    return iter(self._inner_dict)

  def __len__(self):
    return len(self._inner_dict)


class RASPExpr(abc.ABC):
  """A class distinguishing RASP expressions from other objects."""
  _ids = itertools.count(1)

  def __init__(self):
    self._annotations: Mapping[str, Any] = _Annotations(self)

  @abc.abstractmethod
  def __call__(self,
               xs: Sequence[Value]) -> Union[Sequence[Value], SelectorValue]:
    """Evaluates the RASPExpr using the standard evaluator."""

  @property
  def annotations(self) -> Mapping[str, Any]:
    """The annotations of this expression instance."""
    return self._annotations

  @annotations.setter
  def annotations(self, annotations: Mapping[str, Any]):
    self._annotations = _Annotations(self, **annotations)

  @property
  def name(self) -> str:
    """The name of this expression."""
    return self.annotations[_NAME_KEY]

  @property
  @abc.abstractmethod
  def children(self) -> Sequence["RASPExpr"]:
    """Direct dependencies of this expression."""

  @functools.cached_property
  def unique_id(self):
    """A unique id for every expression instance."""
    return next(self._ids)

  def copy(self: RASPExprT) -> RASPExprT:
    """Returns a shallow copy of this RASPExpr with a new ID."""
    return copy.copy(self)

  @property
  def label(self) -> str:
    return f"{self.name}_{self.unique_id}"

  def named(self: RASPExprT, name: str) -> RASPExprT:
    """Convenience method for adding a name."""
    return annotate(self, name=name)

  def annotated(self: RASPExprT, **annotations) -> RASPExprT:
    """Convenience method for adding annotations."""
    return annotate(self, **annotations)


def annotate(expr: RASPExprT, **annotations) -> RASPExprT:
  """Creates a new expr with added annotations."""
  new = expr.copy()
  # Note that new annotations will overwrite existing ones with matching keys.
  new.annotations = {**expr.annotations, **annotations}
  return new


### S-Ops.


class SOp(RASPExpr):
  """A Sequence Operation."""

  def __call__(self, xs: Sequence[Value]) -> Sequence[Value]:
    return evaluate(self, xs)  # pytype: disable=bad-return-type

  # Allow construction of SOps using numeric operators with constant values.
  # Note: if inheriting SOp by a dataclass, make sure to disable eq and order,
  # as they will override these.

  def __lt__(self, other: Value) -> "SOp":
    """self < other."""
    return Map(lambda x: x < other, self)

  def __le__(self, other: Value) -> "SOp":
    """self <= other."""
    return Map(lambda x: x <= other, self)

  def __eq__(self, other: Value) -> "SOp":
    """self == other."""
    return Map(lambda x: x == other, self)

  def __ne__(self, other: Value) -> "SOp":
    """self != other."""
    return Map(lambda x: x != other, self)

  def __gt__(self, other: Value) -> "SOp":
    """self > other."""
    return Map(lambda x: x > other, self)

  def __ge__(self, other: Value) -> "SOp":
    """self >= other."""
    return Map(lambda x: x >= other, self)

  def __add__(self, other: Union["SOp", Value]) -> "SOp":
    """self + other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x + y, self, other)
    return Map(lambda x: x + other, self)

  def __radd__(self, other: Union["SOp", Value]) -> "SOp":
    """other + self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x + y, other, self)
    return Map(lambda x: other + x, self)

  def __sub__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """self - other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x - y, self, other)
    return Map(lambda x: x - other, self)

  def __rsub__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """other - self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x - y, other, self)
    return Map(lambda x: other - x, self)

  def __mul__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """self * other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x * y, self, other)
    return Map(lambda x: x * other, self)

  def __rmul__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """other * self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x * y, other, self)
    return Map(lambda x: other * x, self)

  def __truediv__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """self / other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x / y, self, other)
    return Map(lambda x: x / other, self)

  def __rtruediv__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """other / self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x / y, other, self)
    return Map(lambda x: other / x, self)

  def __invert__(self) -> "SOp":
    return Map(lambda x: not x, self)

  def __and__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """self & other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x and y, self, other)
    return Map(lambda x: x and other, self)

  def __or__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """self | other."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x or y, self, other)
    return Map(lambda x: x or other, self)

  def __rand__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """other & self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x and y, other, self)
    return Map(lambda x: other and x, self)

  def __ror__(self, other: Union["SOp", NumericValue]) -> "SOp":
    """other | self."""
    if isinstance(other, SOp):
      return SequenceMap(lambda x, y: x or y, other, self)
    return Map(lambda x: x or other, self)


class TokensType(SOp):
  """Primitive SOp returning the original input tokens."""

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []

  @property
  def label(self) -> str:
    return "tokens"

  def __repr__(self):
    return "tokens"


class IndicesType(SOp):
  """Primitive SOp returning the position index at each token."""

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []

  @property
  def label(self) -> str:
    return "indices"

  def __repr__(self):
    return "indices"


class LengthType(SOp):
  """Primitive SOp returning the total length of the input."""

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []

  @property
  def label(self) -> str:
    return "length"

  def __repr__(self):
    return "length"


tokens = TokensType()
indices = IndicesType()
length = LengthType()


class Map(SOp):
  """SOp that evaluates the function elementwise on the input SOp.

  Map(lambda x: x + 1, tokens).eval([1, 2, 3]) == [2, 3, 4]
  """

  def __init__(
      self,
      f: Callable[[Value], Value],
      inner: SOp,
      simplify: bool = True,
  ):
    """Initialises.

    Args:
      f: the function to apply elementwise.
      inner: the SOp to which to apply `f`.
      simplify: if True and if `inner` is also a Map, will combine the new map
        and `inner` into a single Map object.
    """
    super().__init__()
    self.f = f
    self.inner = inner

    assert isinstance(self.inner, SOp)
    assert callable(self.f) and not isinstance(self.f, RASPExpr)

    if simplify and isinstance(self.inner, Map):
      # Combine the functions into just one.
      inner_f = self.inner.f
      self.f = lambda t: f(inner_f(t))
      self.inner = self.inner.inner

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.inner]


class SequenceMap(SOp):
  """SOp that evaluates the function elementwise on the two given SOp's.

  SequenceMap(lambda x, y: x - y, length, tokens).eval([1, 2, 3]) == [2, 1, 0]
  """

  def __init__(
      self,
      f: Callable[[Value, Value], Value],
      fst: SOp,
      snd: SOp,
  ):
    super().__init__()

    if fst is snd:
      logging.warning("Creating a SequenceMap with both inputs being the same "
                      "SOp is discouraged. You should use a Map instead.")

    self.f = f
    self.fst = fst
    self.snd = snd
    assert isinstance(self.fst, SOp)
    assert isinstance(self.snd, SOp)
    assert callable(self.f) and not isinstance(self.f, RASPExpr)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.fst, self.snd]


class LinearSequenceMap(SequenceMap):
  """SOp that evaluates a linear function elementwise on the two given SOp's."""

  def __init__(self, fst: SOp, snd: SOp, fst_fac: float, snd_fac: float):
    super().__init__(fst=fst, snd=snd, f=lambda x, y: fst_fac * x + snd_fac * y)
    self.fst_fac = fst_fac
    self.snd_fac = snd_fac


class Full(SOp):
  """A SOp evaluating to [fill]*len(input_values)."""

  def __init__(self, fill: Value):
    super().__init__()
    self.fill = fill

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []


def sop_not(sop: SOp) -> SOp:
  return Map(lambda t: not t, sop)


class ConstantSOp(SOp, Generic[VT]):
  """A constant S-Op for testing purposes."""

  def __init__(self, value: Sequence[VT], check_length: bool = True):
    super().__init__()
    self.value = value
    self.check_length = check_length

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []


### Selectors.


class Predicate(Protocol):

  def __call__(self, key: Value, query: Value) -> bool:
    """Applies the predicate."""


class Comparison(enum.Enum):
  """A two-place boolean comparison predicate for use in Select."""
  EQ = "=="
  LT = "<"
  LEQ = "<="
  GT = ">"
  GEQ = ">="
  NEQ = "!="
  TRUE = "True"
  FALSE = "False"

  def __call__(self, key: Value, query: Value) -> bool:
    if key is None:
      raise ValueError("key is None!")
    if query is None:
      raise ValueError("query is None!")
    return _comparison_table[self](key, query)


_comparison_table = {
    Comparison.EQ: lambda key, query: key == query,
    Comparison.LT: lambda key, query: key < query,
    Comparison.LEQ: lambda key, query: key <= query,
    Comparison.GT: lambda key, query: key > query,
    Comparison.GEQ: lambda key, query: key >= query,
    Comparison.NEQ: lambda key, query: key != query,
    Comparison.TRUE: lambda key, query: True,
    Comparison.FALSE: lambda key, query: False,
}


class Selector(RASPExpr):
  """RASP Selector. Represents something like an attention head's weights."""

  def __call__(self, xs: Sequence[Value]) -> SelectorValue:
    return evaluate(self, xs)  # pytype: disable=bad-return-type

  # Allow construction of Selector combinations using Python logical operators.
  def __and__(self, other: "Selector") -> "Selector":
    """self & other."""
    return selector_and(self, other)

  def __rand__(self, other: "Selector") -> "Selector":
    """other & self."""
    return selector_and(other, self)

  def __or__(self, other: "Selector") -> "Selector":
    """self | other."""
    return selector_or(self, other)

  def __ror__(self, other: "Selector") -> "Selector":
    """other | self."""
    return selector_or(other, self)

  def __invert__(self) -> "Selector":
    """~self."""
    return selector_not(self)


class Select(Selector):
  """Primitive that creates a Selector."""

  def __init__(self, keys: SOp, queries: SOp, predicate: Predicate):
    super().__init__()
    self.keys = keys
    self.queries = queries
    self.predicate = predicate
    assert isinstance(self.keys, SOp)
    assert isinstance(self.queries, SOp)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.keys, self.queries]


class ConstantSelector(Selector):
  """A constant selector for testing purposes."""

  def __init__(self, value: SelectorValue, check_length: bool = True):
    super().__init__()
    self.value = value
    self.check_length = check_length

  @property
  def children(self) -> Sequence[RASPExpr]:
    return []


class SelectorWidth(SOp):
  """SelectorWidth primitive."""

  def __init__(self, selector: Selector):
    super().__init__()
    self.selector = selector
    assert isinstance(self.selector, Selector)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.selector]


class SelectorAnd(Selector):
  """Implements elementwise `and` between selectors."""

  def __init__(self, fst: Selector, snd: Selector):
    super().__init__()
    self.fst = fst
    self.snd = snd
    assert isinstance(self.fst, Selector)
    assert isinstance(self.snd, Selector)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.fst, self.snd]


class SelectorOr(Selector):
  """Implements elementwise `or` between selectors."""

  def __init__(self, fst: Selector, snd: Selector):
    super().__init__()
    self.fst = fst
    self.snd = snd
    assert isinstance(self.fst, Selector)
    assert isinstance(self.snd, Selector)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.fst, self.snd]


class SelectorNot(Selector):
  """Implements elementwise `not` on a selector."""

  def __init__(self, inner: Selector):
    self.inner = inner
    super().__init__()
    assert isinstance(self.inner, Selector)

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.inner]


def selector_not(
    inner: Selector,
    simplify: bool = True,
) -> Selector:
  """Returns a SelectorNot, or a Select if simplifying is possible."""
  if simplify and isinstance(inner, Select):
    predicate = lambda k, q: not inner.predicate(k, q)
    return Select(inner.keys, inner.queries, predicate=predicate)

  return SelectorNot(inner)


def selector_and(
    fst: Selector,
    snd: Selector,
    simplify: bool = True,
) -> Selector:
  """Returns a SelectorAnd, or a Select if simplifying is possible."""
  if simplify and isinstance(fst, Select) and isinstance(snd, Select):
    simplified = _attempt_simplify(fst, snd, lambda l, r: l and r)
    if simplified:
      return simplified

  return SelectorAnd(fst, snd)


def selector_or(
    fst: Selector,
    snd: Selector,
    simplify: bool = True,
) -> Selector:
  """Returns a SelectorOr, or a Select if simplifying is possible."""
  if simplify and isinstance(fst, Select) and isinstance(snd, Select):
    simplified = _attempt_simplify(fst, snd, lambda l, r: l or r)
    if simplified:
      return simplified

  return SelectorOr(fst, snd)


def _attempt_simplify(
    fst: Select,
    snd: Select,
    combine: Callable[[bool, bool], bool],
) -> Optional[Select]:
  """Simplifies two Selects if possible.

  If two Selects in a compound Selector have matching keys and queries, they can
  be simplified into one Select with a compound predicate:

  lambda k,q: combine(fst.predicate(k,q), snd.predicate(k,q))

  This function returns a Select with this predicate if possible,
  and None otherwise.

  A Full SOp in a key or query position is a special case that always matches
  any SOp in the corresponding position in the other selector. In that case,
  we bake in the fill value into the corresponding Select's predicate before
  combining. This allows us to use the other SOp as the input to the simplified
  Select.

  Args:
    fst: the first Select.
    snd: the second Select.
    combine: how to combine the outputs of the individual predicates.

  Returns:
    A combined Select, if possible.
  """
  fst_predicate = fst.predicate
  snd_predicate = snd.predicate
  common_keys = None
  common_queries = None

  if isinstance(fst.keys, Full):
    common_keys = snd.keys
    # We pass the predicate in as a default arg to avoid unintended recursion.
    fst_predicate = lambda key, query, p=fst_predicate: p(fst.keys.fill, query)
  if isinstance(snd.keys, Full):
    common_keys = fst.keys
    snd_predicate = lambda key, query, p=snd_predicate: p(snd.keys.fill, query)
  if isinstance(fst.queries, Full):
    common_queries = snd.queries
    fst_predicate = lambda key, query, p=fst_predicate: p(key, fst.queries.fill)
  if isinstance(snd.queries, Full):
    common_queries = fst.queries
    snd_predicate = lambda key, query, p=snd_predicate: p(key, snd.queries.fill)
  if fst.keys is snd.keys:
    common_keys = fst.keys
  if fst.queries is snd.queries:
    common_queries = fst.queries

  if not common_keys or not common_queries:
    return None

  def predicate(key, query):
    return combine(fst_predicate(key, query), snd_predicate(key, query))

  return Select(common_keys, common_queries, predicate=predicate)


class Aggregate(SOp, Generic[VT]):
  """Aggregate primitive."""

  def __init__(self,
               selector: Selector,
               sop: SOp,
               default: Optional[VT] = None):
    """Initialises. The default is used where nothing is selected."""
    super().__init__()
    self.selector = selector
    self.sop = sop
    self.default = default
    assert isinstance(self.selector, Selector)
    assert isinstance(self.sop, SOp)
    assert (self.default is None or isinstance(self.default,
                                               (str, float, bool, int)))

  @property
  def children(self) -> Sequence[RASPExpr]:
    return [self.selector, self.sop]


### SOp encodings.


class Encoding(enum.Enum):
  """The encoding used by a SOp. Only number-valued SOps support numerical."""
  CATEGORICAL = "categorical"
  NUMERICAL = "numerical"


def numerical(sop: SOpT) -> SOpT:
  return annotate(sop, encoding=Encoding.NUMERICAL)


def categorical(sop: SOpT) -> SOpT:
  return annotate(sop, encoding=Encoding.CATEGORICAL)


def get_encoding(sop: SOp) -> Encoding:
  return sop.annotations["encoding"]


def is_numerical(sop: SOp) -> bool:
  """Check if the SOp is numerically encoded."""
  return get_encoding(sop) == Encoding.NUMERICAL


def is_categorical(sop: SOp) -> bool:
  """Check if the SOp is categorically encoded."""
  return get_encoding(sop) == Encoding.CATEGORICAL


def default_encoding(expr: RASPExpr) -> Optional[Encoding]:
  """Adds an 'encoding' annotation, default is Categorical."""
  if not isinstance(expr, SOp):
    raise TypeError(f"expr {expr} is not a SOp.")

  return Encoding.CATEGORICAL


DEFAULT_ANNOTATORS[_ENCODING_KEY] = default_encoding

### naming.

# Subclasses must appear here before superclasses in order for
# the most specific entry to be used.

_default_name_by_class = {
    # Primitives
    TokensType: "tokens",
    IndicesType: "indices",
    LengthType: "length",
    # SOps
    LinearSequenceMap: "linear_sequence_map",
    SequenceMap: "sequence_map",
    Map: "map",
    Full: "full",
    ConstantSOp: "constant_sop",
    SelectorWidth: "selector_width",
    Aggregate: "aggregate",
    SOp: "sop",
    # Selectors
    Select: "select",
    SelectorAnd: "selector_and",
    SelectorOr: "selector_or",
    SelectorNot: "selector_not",
    ConstantSelector: "constant_selector",
    Selector: "selector",
}


def default_name(expr: RASPExpr) -> Dict[str, str]:
  for cls, name in _default_name_by_class.items():
    if isinstance(expr, cls):
      return name

  raise NotImplementedError(f"{expr} was not given a default name!")


DEFAULT_ANNOTATORS[_NAME_KEY] = default_name

### evaluation.


class RASPEvaluator(abc.ABC):
  """ABC for RASP evaluators."""

  @abc.abstractmethod
  def evaluate(self, expr: RASPExpr,
               xs: Sequence[Value]) -> Union[Sequence[Value], SelectorValue]:
    """Evaluates the RASP expression on input `xs`."""


class DefaultRASPEvaluator(abc.ABC):
  """Default evaluator for RASP."""

  def evaluate(self, expr: RASPExpr,
               xs: Sequence[Value]) -> Union[Sequence[Value], SelectorValue]:
    """Evaluates the RASP expression on input `xs`."""
    return self._eval_fn_by_expr_type[type(expr)](expr, xs)

  def __init__(self):
    self._eval_fn_by_expr_type = {
        # Primitives
        TokensType: self.eval_tokens,
        IndicesType: self.eval_indices,
        LengthType: self.eval_length,
        # SOps
        LinearSequenceMap: self.eval_sequence_map,
        SequenceMap: self.eval_sequence_map,
        Map: self.eval_map,
        Full: self.eval_full,
        ConstantSOp: self.eval_constant_sop,
        SelectorWidth: self.eval_selector_width,
        Aggregate: self.eval_aggregate,
        SOp: _raise_not_implemented,
        # Selectors
        Select: self.eval_select,
        SelectorAnd: self.eval_selector_and,
        SelectorOr: self.eval_selector_or,
        SelectorNot: self.eval_selector_not,
        ConstantSelector: self.eval_constant_selector,
        Selector: _raise_not_implemented,
    }

  def eval_tokens(self, sop: TokensType,
                  xs: Sequence[Value]) -> Sequence[Value]:
    del sop
    return list(xs)

  def eval_indices(self, sop: IndicesType,
                   xs: Sequence[Value]) -> Sequence[Value]:
    del sop
    return list(range(len(xs)))

  def eval_length(self, sop: LengthType, xs: Sequence[Value]) -> Sequence[int]:
    del sop
    return [len(xs)] * len(xs)

  def eval_sequence_map(self, sop: SequenceMap,
                        xs: Sequence[Value]) -> Sequence[Value]:
    fst_values = self.evaluate(sop.fst, xs)
    snd_values = self.evaluate(sop.snd, xs)
    return [
        sop.f(x, y) if None not in [x, y] else None
        for x, y in zip(fst_values, snd_values)
    ]

  def eval_map(self, sop: Map, xs: Sequence[Value]) -> Sequence[Value]:
    return [
        sop.f(x) if x is not None else None
        for x in self.evaluate(sop.inner, xs)
    ]

  def eval_full(self, sop: Full, xs: Sequence[Value]) -> Sequence[Value]:
    return [sop.fill] * len(xs)

  def eval_constant_sop(self, sop: ConstantSOp,
                        xs: Sequence[Value]) -> Sequence[Value]:
    if sop.check_length and (len(xs) != len(sop.value)):
      raise ValueError(
          f"Constant len {len(sop.value)} doesn't match input len {len(xs)}.")
    return sop.value

  def eval_selector_width(self, sop: SelectorWidth,
                          xs: Sequence[Value]) -> Sequence[Value]:
    selector_values = self.evaluate(sop.selector, xs)
    return [sum(row) for row in selector_values]

  def eval_aggregate(self, sop: Aggregate,
                     xs: Sequence[Value]) -> Sequence[Value]:
    selector_value = self.evaluate(sop.selector, xs)
    values = self.evaluate(sop.sop, xs)
    default = sop.default

    return [
        _mean(_get_selected(row, values), default) for row in selector_value
    ]

  def eval_select(self, sel: Select, xs: Sequence[Value]) -> SelectorValue:
    """Evaluates a Select on `xs`."""
    key_values = self.evaluate(sel.keys, xs)
    query_values = self.evaluate(sel.queries, xs)

    key_len = len(key_values)
    query_len = len(query_values)
    out = np.zeros((query_len, key_len), dtype=bool).tolist()
    for row, query in enumerate(query_values):
      for col, key in enumerate(key_values):
        out[row][col] = bool(sel.predicate(key, query))
    return out

  def eval_constant_selector(self, sel: ConstantSelector,
                             xs: Sequence[Value]) -> SelectorValue:
    if sel.check_length and (len(xs) != len(sel.value)):
      raise ValueError(
          f"Constant len {len(xs)} doesn't match input len {len(sel.value)}.")
    return sel.value

  def eval_selector_and(self, sel: SelectorAnd,
                        xs: Sequence[Value]) -> SelectorValue:
    fst_values = self.evaluate(sel.fst, xs)
    snd_values = self.evaluate(sel.snd, xs)
    return np.logical_and(np.array(fst_values), np.array(snd_values)).tolist()

  def eval_selector_or(self, sel: SelectorOr,
                       xs: Sequence[Value]) -> SelectorValue:
    fst_values = self.evaluate(sel.fst, xs)
    snd_values = self.evaluate(sel.snd, xs)
    return np.logical_or(np.array(fst_values), np.array(snd_values)).tolist()

  def eval_selector_not(self, sel: SelectorNot,
                        xs: Sequence[Value]) -> SelectorValue:
    values = self.evaluate(sel.inner, xs)
    return np.logical_not(np.array(values)).tolist()


def _get_selected(
    selector_row: List[bool],
    values: Sequence[VT],
) -> Sequence[VT]:
  """Helper for aggregate. [T T F], [a b c] -> [a b]."""
  return [v for s, v in zip(selector_row, values) if s]


def _mean(xs: Sequence[VT], default: VT) -> VT:
  """Takes the mean for numbers."""
  if not xs:
    return default
  elif len(xs) == 1:
    return xs[0]
  elif all(isinstance(x, (int, bool, float)) for x in xs):
    return sum(xs) / len(xs)
  else:
    raise ValueError(
        "Only types int, bool, and float are supported for aggregation. "
        f"Received sequence: {xs}"
    )


def _raise_not_implemented(expr: RASPExpr, xs: Sequence[Value]):
  raise NotImplementedError(f"Evaluation of {expr} is not defined.")


evaluate = DefaultRASPEvaluator().evaluate

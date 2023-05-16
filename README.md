# Tracr: TRAnsformer Compiler for RASP.

Tracr is a compiler for converting RASP programs
([Weiss et al. 2021](https://arxiv.org/abs/2106.06981))
into transformer weights. Please see our
[tech report](https://arxiv.org/abs/2301.05062) for a detailed description of
the compiler.

Directory structure:

* `rasp` contains an implementation of RASP embedded in Python.
* `compiler` contains the compiler itself.
* `transformer` contains the implementation of the transformer.
* `craft` contains the intermediate representation used by the compiler:
  essentially a small linear algebra-based library with named dimensions.

This is not an officially supported Google product.


## Installation

Just clone and pip install:

```
git clone https://github.com/deepmind/tracr
cd tracr
pip3 install .
```


## Usage example: RASP `reverse` program

Consider the RASP `reverse` program:

```
opp_index = length - indices - 1;
flip = select(indices, opp_index, ==);
reverse = aggregate(flip, tokens);
```

To compile this with Tracr, we would first implement the program using Tracr's
RASP library:

```python
from tracr.rasp import rasp

length = make_length()  # `length` is not a primitive in our implementation.
opp_index = length - rasp.indices - 1
flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
reverse = rasp.Aggregate(flip, rasp.tokens)
```

Where:

```python
def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)
```

We can then compile the RASP program to a transformer with:

```python
from tracr.compiler import compiling

bos = "BOS"
model = compiling.compile_rasp_to_model(
    reverse,
    vocab={1, 2, 3},
    max_seq_len=5,
    compiler_bos=bos,
)
```

This yields a transformer as a [Haiku](https://github.com/deepmind/dm-haiku) model.
This model isn't intended to provide _everything_ you might need, but rather serves
as a kind of "documentation-in-code" for the semantics of the generated parameters.
The expectation is that the user can then write or contribute an adapter that converts
parameters from this reference model to another transformer implementation.

Using this model we can perform a forward pass:

```python
>>> out = model.apply([bos, 1, 2, 3])
>>> out.decoded
["BOS", 3, 2, 1]
```

Success! We have a transformer that reverses its input tokens.

Note: compiled models always expect a BOS token in order to support
selectors which don't attend to any of the input tokens. This is necessary to
preserve intuitive RASP semantics; the alternative would have been to treat
all-False selector rows as equivalent to all-True (which is what softmax in an
attention layer would naturally do). For more details, see our paper.

You can also inspect some of the intermediate activations of the model, using
`out.residuals`, `out.layer_outputs`, and `out.attn_logits`.

For more examples of RASP programs we can compile, check out
[compiler/lib.py](tracr/compiler/lib.py).

For an interactive example of compiling a model and visualizing its computation,
check out the notebook at
[examples/Visualize\_Tracr\_Models.ipynb](tracr/examples/Visualize_Tracr_Models.ipynb).


## Developer README

If you'd like to extend Tracr to fit your purposes, here's some information on
how Tracr works under the hood.


### How Tracr works conceptually

To compile a program, Tracr does the following.

1. **Trace RASP program into a graph representation.** This involves creating
   a graph node for each RASP expression and inferring dependencies between
   these graph nodes.

2. **Infer bases.** Tracr is designed to have each node output to a separate
   subspace of the residual stream. To do this, we first infer the set of all
   possible token values that each node can take, then using that information,
   decide on a subspace for each node, and augment each node in the graph
   with the basis vectors for that node's subspace.

3. **Convert nodes to Craft components.** Craft is the name of our internal
   intermediate representation that does linear algebra on named subspaces. In
   this stage, each expression node is converted to a Craft component that
   actually performs the linear algebra operations necessary to implement the
   expression. This includes converting _sequence operators_ to MLP weights,
   and _selectors_ to weights of attention heads. (We compute the appropriate
   weights directly using the theory of universal approximation for MLPs - no
   gradient descent required!)

4. **Convert Craft graph to Craft model.** In this stage, we convert from
   a graph representation to a layout that looks more like an actual
   transformer. At this stage, we essentially have a working model, but
   with the linear algebra done using Craft rather than JAX + Haiku.

5. **Convert Craft model to Haiku model.** Finally, we convert our
   intermediate representation of the model to a full Haiku model.

Two details worth expanding on here are subspaces and corresponding bases.
Each node writes to a separate subspace of the residual stream,
where each subspace is simply a unique chunk of the residual stream vector.
For example, the first node might write to the first 5 components of
the residual stream; the second node the next 5; and so on.  In terms of what
the embeddings actually associated with each node, Tracr employs two
different kinds of bases:

* **Categorical representation** - in which each unique token value is
  represented as a unique one-hot vector in that node's subspace. This
  is the representation used by default.
* **Numerical representation** - in which each unique token value is
  mapped to a unique scalar value. This is necessary for some uses
  of the `aggregate` operation - essentially, ones which involve taking
  a mean - and some other operations are represented more efficiently
  with this representation.

A final detail is BOS tokens. The compiler relies on beginning-of-sequence
tokens to in order to implement a number of operations. This is why token
sequences fed into the final model _must_ start with a BOS token.


### How Tracr works in practice

The flow of compilation execution begins in
[`compiler/compiling.py`](tracr/compiler/compiling.py), in the
`compile_rasp_to_model` function. This function is fairly short and maps
directly to the stages outlined above, so don't be afraid to read the source!


### Running tests

We use [`absltest`](https://abseil.io/docs/python/guides/testing), which is
`unittest`-compatible, and is therefore in turn `pytest`-compatible.

First, install test dependencies:

```
pip3 install absl-py pytest
```

Then, in the checkout directory, simply run `pytest`. This should take about 60
seconds.


## Superposition

One topic that we've investigated using Tracr is superposition (see e.g.
[Elhage et al 2023](https://transformer-circuits.pub/2022/toy_model/index.html)):
in this work, we learn a compressed embedding of the residual stream in such a
way as to keep the computation faithful to the uncompressed program.

This is an example showing the dot products between embedding vectors for the
`frac_prevs` example program from [compiler/lib.py](tracr/compiler/lib.py):

![Matrix of dot-products](compression_heatmap.png)

The code for learning these embeddings is not included in this repository, but
you can read more about it in Section 5 of the
[tech report](https://arxiv.org/abs/2301.05062).


## Citing Tracr

Please use the bibtex for our tech report:

```
@article{lindner2023tracr,
  title = {Tracr: Compiled Transformers as a Laboratory for Interpretability},
  author = {Lindner, David and Kramár, János and Rahtz, Matthew and McGrath, Thomas and Mikulik, Vladimir},
  journal={arXiv preprint arXiv:2301.05062},
  year={2023}
}
```

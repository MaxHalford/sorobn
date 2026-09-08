import inspect

import streamlit as st

import sorobn

"""
# 🦔 sorobn

This is a little demo app for the [sorobn library](https://github.com/MaxHalford/sorobn).
"""

examples = dict(inspect.getmembers(sorobn.examples, inspect.isfunction))

example = st.selectbox("Pick a network", list(examples.keys()))

bn = examples[example]()

st.graphviz_chart(bn.graphviz())

"""
## Conditional probability tables
"""

var = st.selectbox("Select a variable", bn.nodes)
cpt = bn.P[var].to_frame().reset_index()
cpt

"""
## Inference
"""

target_vars = st.multiselect("Target variables", bn.nodes, default=bn.nodes[:1])

given_vars = st.multiselect("Evidence variables", bn.nodes, default=bn.nodes[1:2])

if target_vars:
    """Posterior"""
    answer = bn.distribution(
        *target_vars, given={var: True for var in given_vars}, algorithm="exact"
    )
    answer = answer.to_frame().reset_index()
    answer

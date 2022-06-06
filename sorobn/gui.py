import inspect

import sorobn as hh
import streamlit as st

"""
# 🦔 sorobn

This is a little demo app for the [sorobn library](https://github.com/MaxHalford/sorobn).
"""

examples = dict(inspect.getmembers(hh.examples, inspect.isfunction))

example = st.selectbox('Pick a network', list(examples.keys()))

bn = examples[example]()

st.graphviz_chart(bn.graphviz())

"""
## Conditional probability tables
"""

var = st.selectbox('Select a variable', bn.nodes)
cpt = bn.P[var].to_frame().reset_index()
cpt

"""
## Inference
"""

query_vars = st.multiselect('Query variables', bn.nodes, default=bn.nodes[:1])

events_vars = st.multiselect('Event variables', bn.nodes, default=bn.nodes[1:2])

if query_vars:
    """Posterior"""
    answer = bn.query(*query_vars, event={var: True for var in events_vars}, algorithm='exact')
    answer = answer.to_frame().reset_index()
    answer

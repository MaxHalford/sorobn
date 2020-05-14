import hedgehog as hh
import streamlit as st

"""
# 🦔 hedgehog

This is a little demo app for the [hedgehog library](https://github.com/MaxHalford/hedgehog).
"""

loaders = {
    l.split('_', 1)[1].capitalize(): eval(f'hh.{l}') for l in hh.__all__
    if l.startswith('load_')
}

loader = st.selectbox('Pick a network', list(loaders.keys()))

bn = loaders[loader]()

st.graphviz_chart(bn.graphviz())

"""
## Conditional probability tables
"""

var = st.selectbox('Select a variable', bn.nodes)
cpt = bn.cpts[var].to_frame().reset_index()
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

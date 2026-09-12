Graph tools
===========

.. currentmodule:: pytensor_ml.pytensorf

Compilation
-----------

.. autosummary::
    :toctree: generated/

    function
    compile_predict
    rewrite_for_prediction
    rewrite_pregrad

Graph inspection
----------------

.. autosummary::
    :toctree: generated/

    collect_graph_inputs
    collect_data_inputs
    collect_trainable_params
    collect_non_trainable_params
    collect_differentiable_params
    collect_shared_variables
    collect_step_counters
    collect_clock_updates
    collect_non_trainable_updates
    find_rng_nodes
    as_output_list

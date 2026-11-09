"""Console wrappers for the historical H3 commands, preserving their recipe."""


def legacy_uni():
    from .train_lora_minimax_h3 import main
    main()  # The historical callable returns a result dict, not an exit code.


def legacy_diag():
    from .diag_minimax_h3_uni import main
    main()

import sys
import cnn_asm_transfer_train


if __name__ =='__main__':
    # pretrain_probs = ["", "SUBINC", "SUMTRIAN"]
    # probs = ["FLOW016"]

    # pretrain_probs = ["FLOW016", "SUBINC", "SUMTRIAN"]
    # probs = ["MNMX"]

    # pretrain_probs = ["FLOW016", "MNMX", "SUMTRIAN"]
    # probs = ["SUBINC"]

    pretrain_probs = ["FLOW016", "MNMX", "SUBINC"]
    probs = ["SUMTRIAN"]

    for p in probs:
        print ("run training on ", p)
        cnn_asm_transfer_train.run_net(pretrain_probs=pretrain_probs, probs=[p])

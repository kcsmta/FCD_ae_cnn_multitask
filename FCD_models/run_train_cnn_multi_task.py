import sys
import ae_cnn_asm_multi_task_NoPretrain_train


if __name__ =='__main__':
    # all probs = ["FLOW016", "MNMX", "SUBINC", "SUMTRIAN"]
    # probs =["debug"] # for testing implementation
    probs = ["FLOW016", "MNMX", "SUBINC", "SUMTRIAN"]
    for p in probs:
        print ("run training on ", p)
        ae_cnn_asm_multi_task_NoPretrain_train.run_net(probs=[p])

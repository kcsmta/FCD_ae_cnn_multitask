import sys
import ae_cnn_asm_multi_task_train


if __name__ =='__main__':
    # all probs = ["FLOW016", "MNMX", "SUBINC", "SUMTRIAN"]
    probs = ["FLOW016", "MNMX", "SUBINC", "SUMTRIAN"]
    for p in probs:
        print ("run training on ", p)
        ae_cnn_asm_multi_task_train.run_net(probs=[p])

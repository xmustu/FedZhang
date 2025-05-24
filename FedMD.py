import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer


def pearson_corr(x, y, axis):
    """
    计算 x, y 在指定 axis 上的 Pearson 相关系数：
      - x, y: 张量，shape=(..., D, ...)
      - axis:  求相关的维度（1 表示对每个样本的 D 类别维度相关；0 表示对每个类别的样本维度相关）
    返回：沿 axis 后的相关值张量。
    """
    x_cent = x - tf.reduce_mean(x, axis=axis, keepdims=True)
    y_cent = y - tf.reduce_mean(y, axis=axis, keepdims=True)
    num = tf.reduce_sum(x_cent * y_cent, axis=axis)
    den = tf.norm(x_cent, axis=axis) * tf.norm(y_cent, axis=axis) + 1e-8
    return num / den

def cosine_similarity(x, y, axis):
    """
    计算 x, y 在指定 axis 上的余弦相似度：
      - x, y: 张量，shape=(..., D, ...)
      - axis: 求相似度的维度（1 表示对每个样本的 D 类别维度相似；0 表示对每个类别的样本维度相似）
    返回：沿 axis 后的相似度张量。
    """
    # 计算分子：点积
    num = tf.reduce_sum(x * y, axis=axis)
    
    # 计算分母：范数乘积
    den = tf.norm(x, axis=axis) * tf.norm(y, axis=axis) + 1e-8
    
    # 计算余弦相似度
    cos_sim = num / den
    
    return cos_sim

def inter_intra_loss(beta=1.0, gamma=1.0, tau=1.0):
    """
    返回只计算 L_inter 和 L_intra的复合损失函数，用于 logits 分支：
      输入 y_true_logits, y_pred_logits 均 shape=(batch_size, num_classes)
    L_inter: 样本级相关；L_intra: 类别级相关
    """
    def loss_fn(y_true_logits, y_pred_logits):
        # 温度缩放 logits 后得到概率
        p_s = tf.nn.softmax(y_pred_logits / tau, axis=1)  # shape (B, C)
        p_t = tf.nn.softmax(y_true_logits / tau, axis=1)  # shape (B, C)

        # 类间：对每个样本，在类别维度(1)上计算 Pearson，然后取平均
        corr_inter = cosine_similarity(p_s, p_t, axis=1)        # shape (B,)
        L_inter = tau**2 * (1.0 - tf.reduce_mean(corr_inter))

        # 类内：对每个类别，在样本维度(0)上计算 Pearson，然后取平均
        corr_intra = cosine_similarity(p_s, p_t, axis=0)        # shape (C,)
        L_intra = tau**2 * (1.0 - tf.reduce_mean(corr_intra))

        return beta * L_inter + gamma * L_intra
    return loss_fn

def kl_only_loss(tau=1.0):
    """
    仅基于聚合 logits (y_true) 与客户端 logits (y_pred) 的 KL 散度。
    y_true: 教师平均 logits, shape=(B, C)
    y_pred: 学生输出 logits,   shape=(B, C)
    """
    # 使用 TensorFlow 自带的 KLDivergence
    kld = tf.keras.losses.KLDivergence()
    def loss_fn(y_true, y_pred):
        # 对 logits 进行温度缩放并 softmax
        p_t = tf.nn.softmax(y_true / tau, axis=1)  # 教师概率
        p_s = tf.nn.softmax(y_pred / tau, axis=1)  # 学生概率
        # 计算 KL(p_t || p_s)，注意 KLDivergence 默认计算 KL(y_true ‖ y_pred)
        return kld(p_t, p_s) * (tau ** 2)
    return loss_fn

class FedMD():
    def __init__(self, parties, public_dataset, 
                 private_data, total_private_data,  
                 private_test_data, N_alignment,
                 N_rounds, 
                 N_logits_matching_round, logits_matching_batchsize, 
                 N_private_training_round, private_training_batchsize):
        
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment
        
        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        
        # self.collaborative_parties = []
        self.init_result = []

        # 初始化后产生一个20x10的模型数组
        self.branches = []

        print("start model initialization: ")
        num_branches = 25
        # 按顺序排列如下：
        # DIST：
        # 1.0+1.0+1.0
        # 1.0+1.0+2.0
        # 1.0+1.0+4.0
        # 1.0+1.0+8.0
        # 0.5+2.0+1.0
        # 0.5+2.0+2.0
        # 0.5+2.0+4.0
        # 0.5+2.0+8.0
        # 2.0+0.5+1.0
        # 2.0+0.5+2.0
        # 2.0+0.5+4.0
        # 2.0+0.5+8.0
        # 1.0+0+1.0
        # 1.0+0+2.0
        # 1.0+0+4.0
        # 1.0+0+8.0
        # 0+1.0+1.0
        # 0+1.0+2.0
        # 0+1.0+4.0
        # 0+1.0+8.0
        # KL：0.5、1.0、2.0、4.0、8.0
        for i in range(num_branches):
            branch_models = []
            # for party in parties:
            #     model = clone_model(party)
            #     model.set_weights(party.get_weights())
            #     model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), 
            #                 loss="sparse_categorical_crossentropy",
            #                 metrics=["accuracy"])
            #     branch_models.append({
            #         "model_logits": remove_last_layer(model, loss="mean_absolute_error"),
            #         "model_classifier": model,
            #         "model_weights": model.get_weights()
            #     })
            for i in range(self.N_parties):
                print("model ", i)
                model_A_twin = None
                model_A_twin = clone_model(parties[i])
                model_A_twin.set_weights(parties[i].get_weights())
                model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                                    loss = "sparse_categorical_crossentropy",
                                    metrics = ["accuracy"])
                
                # print("start full stack training ... ")        
                
                model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
                                batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                                validation_data = [private_test_data["X"], private_test_data["y"]],
                                callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                                )
                
                # print("full stack training done")
                
                model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")
                branch_models.append({"model_logits": model_A, 
                                        "model_classifier": model_A_twin,
                                        "model_weights": model_A_twin.get_weights()})
                # self.collaborative_parties.append({"model_logits": model_A, 
                #                                 "model_classifier": model_A_twin,
                #                                 "model_weights": model_A_twin.get_weights()})
                # print()
                del model_A, model_A_twin
            #END FOR LOOP
            self.branches.append(branch_models)

        
        
        print("calculate the theoretical upper bounds for participants: ")
        
        self.upper_bounds = []
        self.pooled_train_result = []
        for model in parties:
            model_ub = clone_model(model)
            model_ub.set_weights(model.get_weights())
            model_ub.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
                             loss = "sparse_categorical_crossentropy", 
                             metrics = ["accuracy"])
            
            model_ub.fit(total_private_data["X"], total_private_data["y"],
                         batch_size = 32, epochs = 50, shuffle=True, verbose = 0, 
                         validation_data = [private_test_data["X"], private_test_data["y"]],
                         callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10)])
            
            self.upper_bounds.append(model_ub.history.history["val_accuracy"][-1])
            self.pooled_train_result.append({"val_acc": model_ub.history.history["val_accuracy"], 
                                             "acc": model_ub.history.history["accuracy"]})
            
            del model_ub    
        with open("ImbalanceGrid", "a", encoding="utf-8") as f:
            print("the upper bounds are:", self.upper_bounds, file=f)
    
    def collaborative_training(self):
        # start collaborating training    
        # collaboration_performance = {i: [] for i in range(self.N_parties)}
        collaboration_performance_set = [
            {i: [] for i in range(self.N_parties)}
            for _ in range(len(self.branches))
        ]
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"], 
                                                     self.public_dataset["y"],
                                                     self.N_alignment)
            with open("ImbalanceGrid", "a", encoding="utf-8") as f:
                print("round ", r, file=f)
            logits = []


            # （待ctrl+F）branch就相当于原来的self.collaborative_parties
            # （待ctrl+F）collaboration_performance_set[branch_index]就相当于原来的collaboration_performance
            # （待ctrl+F）logits[branch_index]就相当于原来的logits
            # 等修改完以后，再把1个复制为5个，另1个复制为20个，并改参数
            for branch_index, branch in enumerate(self.branches):
                # 更新 logits
                # logits = sum([d["model_logits"].predict(alignment_data["X"], verbose=0) for d in branch]) / self.N_parties
                logits.append(0)
                for d in branch:
                    d["model_logits"].set_weights(d["model_weights"])
                    logits[branch_index] += d["model_logits"].predict(alignment_data["X"], verbose = 0)
                logits[branch_index] /= self.N_parties
                # 评估性能
                with open("ImbalanceGrid", "a", encoding="utf-8") as f:
                    print("test performance ... ", (branch_index + 1), file=f)    
                # 初始化浮点数数组用于装结果
                tensor_a = tf.zeros(shape=(0,), dtype=tf.float32)
                for index, d in enumerate(branch):
                    y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                    collaboration_performance_set[branch_index][index].append(np.mean(self.private_test_data["y"] == y_pred))
                    # print(collaboration_performance_set[branch_index][index][-1])
                    normalized_value = collaboration_performance_set[branch_index][index][-1] / self.upper_bounds[index]
                    tensor_a = tf.concat([tensor_a, [normalized_value]], axis=0)
                    del y_pred
                with open("ImbalanceGrid", "a", encoding="utf-8") as f:
                    print("百分比：", tensor_a.numpy(), file=f)
                mean_value = tf.reduce_mean(tensor_a)
                with open("ImbalanceGrid", "a", encoding="utf-8") as f:
                    print("平均值：", mean_value.numpy(), file=f)
                variance_value = tf.math.reduce_variance(tensor_a)
                with open("ImbalanceGrid", "a", encoding="utf-8") as f:
                    print("方差：", variance_value.numpy(), file=f)
                # for index, d in enumerate(branch):
                #     y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                #     collaboration_performance_set[branch_index][index].append(np.mean(self.private_test_data["y"] == y_pred))
                    
                #     print(collaboration_performance_set[branch_index][index][-1])
                #     del y_pred
                
            # 判断是否是最后一轮epoch
            r+= 1
            if r > self.N_rounds:
                return True
            
            # 对每个模型进行对齐和私有训练
            
            # for branch_index, branch in enumerate(self.branches):
            #     for index, d in enumerate(branch):
            #         model = clone_model(d["model_logits"])
            #         model.set_weights(d["model_weights"])
            #         model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            #                     loss=inter_intra_loss(beta=self.beta, gamma=self.gamma, tau=self.tauDIST))
            #         model.fit(alignment_data["X"], logits,
            #                 batch_size=self.logits_matching_batchsize,
            #                 epochs=self.N_logits_matching_round,
            #                 shuffle=True, verbose=0)
            #         d["model_weights"] = model.get_weights()
            #         d["model_logits"].set_weights(d["model_weights"])

            #         # 私有训练
            #         d["model_classifier"].set_weights(d["model_weights"])
            #         d["model_classifier"].fit(self.private_data[index]["X"], 
            #                                 self.private_data[index]["y"],       
            #                                 batch_size=self.private_training_batchsize, 
            #                                 epochs=self.N_private_training_round, 
            #                                 shuffle=True, verbose=0)
            #         d["model_weights"] = d["model_classifier"].get_weights()
  
            # print("updates models ...")
            # for index, d in enumerate(self.collaborative_parties):
            #     print("model {0} starting alignment with public logits... ".format(index))
                
                
            #     # weights_to_use = None
            #     # weights_to_use = d["model_weights"]

            #     # d["model_logits"].set_weights(weights_to_use)
            #     # d["model_logits"].fit(alignment_data["X"], logits, 
            #     #                       batch_size = self.logits_matching_batchsize,  
            #     #                       epochs = self.N_logits_matching_round, 
            #     #                       shuffle=True, verbose = 0)
            #     # d["model_weights"] = d["model_logits"].get_weights()
            #     print("model {0} done alignment".format(index))

            #     print("model {0} starting training with private data... ".format(index))
            #     weights_to_use = None
            #     weights_to_use = d["model_weights"]
            #     d["model_classifier"].set_weights(weights_to_use)
            #     d["model_classifier"].fit(self.private_data[index]["X"], 
            #                               self.private_data[index]["y"],       
            #                               batch_size = self.private_training_batchsize, 
            #                               epochs = self.N_private_training_round, 
            #                               shuffle=True, verbose = 0)

            #     d["model_weights"] = d["model_classifier"].get_weights()
            #     print("model {0} done private training. \n".format(index))
            # #END FOR LOOP


            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 1)
            for index, d in enumerate(self.branches[0]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=1.0, tau=1.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[0],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 2)
            for index, d in enumerate(self.branches[1]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=1.0, tau=2.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[1],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 3)
            for index, d in enumerate(self.branches[2]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=1.0, tau=4.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[2],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 4)
            for index, d in enumerate(self.branches[3]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=1.0, tau=8.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[3],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 5)
            for index, d in enumerate(self.branches[4]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0.5, gamma=2.0, tau=1.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[4],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 6)
            for index, d in enumerate(self.branches[5]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0.5, gamma=2.0, tau=2.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[5],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 7)
            for index, d in enumerate(self.branches[6]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0.5, gamma=2.0, tau=4.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[6],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 8)
            for index, d in enumerate(self.branches[7]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0.5, gamma=2.0, tau=8.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[7],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 9)
            for index, d in enumerate(self.branches[8]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=2.0, gamma=0.5, tau=1.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[8],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 10)
            for index, d in enumerate(self.branches[9]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=2.0, gamma=0.5, tau=2.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[9],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 11)
            for index, d in enumerate(self.branches[10]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=2.0, gamma=0.5, tau=4.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[10],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 12)
            for index, d in enumerate(self.branches[11]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=2.0, gamma=0.5, tau=8.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[11],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 13)
            for index, d in enumerate(self.branches[12]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=0, tau=1.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[12],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 14)
            for index, d in enumerate(self.branches[13]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=0, tau=2.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[13],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 15)
            for index, d in enumerate(self.branches[14]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=0, tau=4.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[14],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 16)
            for index, d in enumerate(self.branches[15]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=1.0, gamma=0, tau=8.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[15],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 17)
            for index, d in enumerate(self.branches[16]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0, gamma=1.0, tau=1.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[16],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 18)
            for index, d in enumerate(self.branches[17]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0, gamma=1.0, tau=2.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[17],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 19)
            for index, d in enumerate(self.branches[18]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0, gamma=1.0, tau=4.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[18],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数beta=self.beta, gamma=self.gamma, tau=self.tauDIST
            print("updates models ...", 20)
            for index, d in enumerate(self.branches[19]):
                # print("dist_model {0} starting alignment with public logits... ".format(index))
                # dist
                model_B = clone_model(d["model_logits"])
                model_B.set_weights(d["model_weights"])
                model_B.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=inter_intra_loss(beta=0, gamma=1.0, tau=8.0)
                )   
                model_B.fit(
                    alignment_data["X"], logits[19],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_B.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("dist_model {0} done alignment".format(index))

                # print("dist_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("dist_model {0} done private training. \n".format(index))
            #END FOR LOOP

            
            
            
            
            
            
            
            
            
            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数tau=self.tauKL
            print("updates models ...", 21)
            for index, d in enumerate(self.branches[20]):
                # print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=kl_only_loss(tau=0.5)
                )   
                model_C.fit(
                    alignment_data["X"], logits[20],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_C.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("kl_model {0} done alignment".format(index))

                # print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数tau=self.tauKL
            print("updates models ...", 22)
            for index, d in enumerate(self.branches[21]):
                # print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=kl_only_loss(tau=1.0)
                )   
                model_C.fit(
                    alignment_data["X"], logits[21],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_C.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("kl_model {0} done alignment".format(index))

                # print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数tau=self.tauKL
            print("updates models ...", 23)
            for index, d in enumerate(self.branches[22]):
                # print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=kl_only_loss(tau=2.0)
                )   
                model_C.fit(
                    alignment_data["X"], logits[22],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_C.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("kl_model {0} done alignment".format(index))

                # print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数tau=self.tauKL
            print("updates models ...", 24)
            for index, d in enumerate(self.branches[23]):
                # print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=kl_only_loss(tau=4.0)
                )   
                model_C.fit(
                    alignment_data["X"], logits[23],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_C.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("kl_model {0} done alignment".format(index))

                # print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP

            # 需要改动self.branches[]、logits[]、print("updates models ...", )、超参数tau=self.tauKL
            print("updates models ...", 25)
            for index, d in enumerate(self.branches[24]):
                # print("kl_model {0} starting alignment with public logits... ".format(index))
                
                # kl
                model_C = clone_model(d["model_logits"])
                model_C.set_weights(d["model_weights"])
                model_C.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=kl_only_loss(tau=8.0)
                )   
                model_C.fit(
                    alignment_data["X"], logits[24],
                    batch_size = self.logits_matching_batchsize,
                    epochs = self.N_logits_matching_round,
                    shuffle=True, verbose = 0
                )
                new_w = model_C.get_weights()
                d["model_weights"] = new_w
                d["model_logits"].set_weights(new_w)
                
                # print("kl_model {0} done alignment".format(index))

                # print("kl_model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"], 
                                          self.private_data[index]["y"],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                d["model_weights"] = d["model_classifier"].get_weights()
                # print("kl_model {0} done private training. \n".format(index))
            #END FOR LOOP


        #END WHILE LOOP
        return True


        
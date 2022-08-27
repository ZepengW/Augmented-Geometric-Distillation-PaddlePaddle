import numpy as np
# fake input
#fake_data = np.random.rand(128, 3, 256, 128).astype(np.float32) - 0.5
# fake layer0 output
#fake_data = np.random.rand(128, 64, 64, 32).astype(np.float32) - 0.5
#fake layer1 output
#fake_data = np.random.rand(128, 256, 64, 32).astype(np.float32) - 0.5
# fake layer2 output
#fake_data = np.random.rand(128, 512, 32, 16).astype(np.float32) - 0.5
# fake layer3 output
#fake_data = np.random.rand(128, 1024, 16, 8).astype(np.float32) - 0.5
#fake layer4 output
#fake_data = np.random.rand(128, 2048, 8, 4).astype(np.float32) - 0.5
#np.save("./data/test_diff/fake_layer0_output.npy", fake_data)

# fake label
#fake_label = np.random.randint(1, 1042, size = 128, dtype = np.int64)
#np.save("./data/test_diff/fake_label.npy", fake_label)
# fake output global
fake_global = np.random.rand(128, 2048).astype(np.float32) - 0.5
# fake output preds
fake_preds = np.random.rand(128, 1041).astype(np.float32) - 0.5
np.save("./data/test_diff/fake_global.npy", fake_global)
np.save("./data/test_diff/fake_preds.npy", fake_preds)
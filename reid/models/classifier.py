import math
import paddle
from paddle import nn
from paddle.nn import functional as F


class MultiBranchClassifier(nn.Layer):
    def __init__(self, **classifiers):
        super(MultiBranchClassifier, self).__init__()
        self.classifiers = nn.LayerDict(classifiers)

        print(self)

    def forward(self, key, embedded_features, labels, **kwargs):

        return self.classifiers[key](embedded_features, labels)

    def reset_w(self, key, groups, memory_bank):
        self.classifiers[key].reset_w(groups, memory_bank)

    def add_classifiers(self, **classifiers):
        self.classifiers.update(classifiers)
        print(f"Update classifiers: {classifiers} \n"
              f"MultiBranchClassifier with classifiers: {self.classifiers} \n")

    def __repr__(self):
        return f"Build MultiBranchClassifier with classifiers: {self.classifiers} \n"


class Linear(nn.Layer):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_tmp = paddle.randn([self.out_features, self.in_features])
        self.W = paddle.create_parameter(shape=w_tmp.shape, dtype=(w_tmp.numpy().dtype),\
            default_initializer=nn.initializer.Assign(w_tmp))
        # nn.init.normal_(self.W, std=0.001)
        normal_m = nn.initializer.Normal(std=0.001)
        normal_m(self.W)
        
        print(self)

    def forward(self, input, *args):
        return input.mm(self.W.t())

    def reset_w(self, groups, memory_bank):
        weights = paddle.randn([len(groups), self.in_features])
        for idx, members in groups.items():
            member_features = memory_bank[list(members)].mean(dim=0)
            weights[idx] = member_features.detach()

        self.W = paddle.create_parameter(shape=weights.shape, dtype=(weights.numpy().dtype),\
            default_initializer=nn.initializer.Assign(weights))

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features},) \n"

    def expand(self, num_classes):
        self.out_features += num_classes
        w_tmp = paddle.randn([num_classes, self.in_features])
        new_W = paddle.create_parameter(shape=w_tmp.shape, dtype=(w_tmp.numpy().dtype),\
            default_initializer=nn.initializer.Assign(w_tmp))
        # nn.init.normal_(new_W, std=0.001)
        normal_m = nn.initializer.Normal(std=0.001)
        normal_m(new_W)

        expanded_W = paddle.cat([new_W.data, self.W.data], dim=0)
        self.W = paddle.create_parameter(shape=expanded_W.shape, dtype=(expanded_W.numpy().dtype),\
            default_initializer=nn.initializer.Assign(expanded_W))
        print(f"Expand W from {self.out_features-num_classes} classes to {self.out_features}")


class Sphere(nn.Layer):
    def __init__(self, in_features, out_features, scale):
        super(Sphere, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

        w_tmp = paddle.randn([self.out_features, self.in_features])
        self.W = paddle.create_parameter(shape=w_tmp.shape, dtype=(w_tmp.numpy().dtype),\
            default_initializer=nn.initializer.Assign(w_tmp))
        normal_m = nn.initializer.Normal(std=0.001)
        normal_m(self.W)

        print(self)

    def forward(self, input, *args):
        input_l2 = input.pow(2).sum(dim=1, keepdim=True).pow(0.5).clamp(min=1e-12)
        input_norm = input / input_l2
        W_l2 = self.W.pow(2).sum(dim=1, keepdim=True).pow(0.5).clamp(min=1e-12)
        W_norm = self.W / W_l2
        cos_th = input_norm.mm(W_norm.t())
        s_cos_th = self.scale * cos_th

        return s_cos_th

    def reset_w(self, groups, memory_bank):
        weights = paddle.randn([len(groups), self.in_features])
        for idx, members in groups.items():
            member_features = memory_bank[list(members)].mean(dim=0)
            weights[idx] = member_features.detach()
        self.W = paddle.create_parameter(shape=weights.shape, dtype=(weights.numpy().dtype),\
            default_initializer=nn.initializer.Assign(weights))

    def __repr__(self):
        return f"Sphere(in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}) \n"


class Arc(nn.Layer):

    def __init__(self, in_features, out_features, scale=20.0, margin=0.1):
        super(Arc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        w_tmp = paddle.randn([self.out_features, self.in_features])
        self.weight = paddle.create_parameter(shape=w_tmp.shape, dtype=(w_tmp.numpy().dtype),\
            default_initializer=nn.initializer.Assign(w_tmp))

        normal_m = nn.initializer.Normal(std=0.001)
        normal_m(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        print(self)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = paddle.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = paddle.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def __repr__(self):
        return f"Arc(in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}), margin={self.margin}. \n"


class NoNormArc(nn.Layer):

    def __init__(self, in_features, out_features, margin=0.06):
        super(NoNormArc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        w_tmp = paddle.randn([self.out_features, self.in_features])
        self.weight = paddle.create_parameter(shape=w_tmp.shape, dtype=(w_tmp.numpy().dtype),\
            default_initializer=nn.initializer.Assign(w_tmp))

        normal_m = nn.initializer.Normal(std=0.001)
        normal_m(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

        print(self)

    def forward(self, x, label):
        x_norm = x.norm(p=2, dim=1, keepdim=True)
        w_norm = self.weight.norm(p=2, dim=1, keepdim=True)

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = paddle.sqrt(1.0 - paddle.pow(cosine, 2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = paddle.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = paddle.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * x_norm * w_norm.t()

        return output

    def __repr__(self):
        return f"NoNormArc(in_features={self.in_features}, out_features={self.out_features}, margin={self.margin}. \n"

# torch autograd

this blogpost is my documented journey through autograd in torch.
for the last two years i've been using pytorch, without properly understanding the mechanics of automatic differentiation. of course i was aware of the mechanism of backprop in general, as we thoroughly learned it in a deep learning course at university.
there we studied it from a mathematical perspective and had to rehearse it until we could do it in our sleep. but i never fully understood how it was actually implemented in pytorch.
and honestly, that was fine. if i would not have touched torch and python's deep learning libraries without knowing every detail of its mechanisms, i would have never started. torch can be overwhelming at first and so much is possible. therefore i'm a big proponent of using it hands on and learning it afterwards when you need it on the fly, à la karpathy ([this tweet](https://x.com/karpathy/status/1325154823856033793?lang=en)).
"you only need to learn on demand".



## mathematical perspective of backpropagation
the general goal of backpropagation is to calculate $\frac{\partial L}{\partial w}$. this is, for every weight $w \in \theta$, we want to calculate its derivative of the loss.
this means intuitively, how much weight $w_i$ contributes to the current loss $L$, to see how you can improve the network.

the core idea is that this can be calculated using the chain rule, which comes in handy. to revisit the chain rule:

\[
    g(f(x))' = g'(f(x)) \cdot f'(x)
\]


the objective for our statistical model $f(x)$ is to minimize the loss, eg.
\[
    \min_\theta L(y, f(x; \theta))
\]

for every single weight $w \in \theta$. so ideally, the following holds: $L(y, f(x;\theta))=0$.

how to do that? deriving the loss in terms of $w$ and updating $w$ accordingly using gradient descent!

therefore, we want to compute
\[
    \frac{\partial L}{\partial w} = \frac{\partial L}{\partial f}  \frac{\partial f}{\partial w}
\]

suppose we use a simple neural network with only one layer with three weights. therefore, the model can be formalized to:
\[
 f(x;\theta) =h(w_1 x_1 + w_2 x_2 + w_3 x_3 + b) = h(a)
\]

note here, that the bias can be handled as an additional weight with constant input of 1, so we can ignore it for now.
here, $h$ is the activation function, e.g. ReLU or sigmoid. for this formalization, let's use the identity function, so $h(a) = a$. in addition, we use $L2$ loss:

\[
L(x,y) = (x-y)^2
\]

to compute $\frac{\partial L}{\partial w_1}$, we get the following:
\[
    \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial w_1}
\]

now we can compute each part separately:
- $\frac{\partial a}{\partial w_1} = x_1$ (simple derivative of linear function)
- $\frac{\partial f}{\partial a} = h'(a) = 1$ (derivative of identity function)
- $\frac{\partial L}{\partial f} = 2(f(x) - y)$ (derivative of L2 loss)

we simply stick this together by multiplying it:

\[
    \frac{\partial L}{\partial w_1} = 2(f(x) - y) \cdot 1 \cdot x_1 = 2(f(x) - y) x_1
\]

of course in real neural networks we use different activation functions. for every activation function however it is possible to calculate its derivative beforehand.
the same holds for different loss functions. it is easy to calculate the derivative of the loss function beforehand.

for deeper networks with multiple layers, the same principle applies. the chain rule is applied multiple times, where each layer contributes its own derivative.

let's say we have a two layer network:
\[
 f(x;\theta) = h_2(w_2 h_1(w_1 x + b_1) + b_2)
\]
to compute $\frac{\partial L}{\partial w_1}$, we get:
\[
    \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial h_2} \cdot \frac{\partial h_2}{\partial a_2} \cdot \frac{\partial a_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}
\]
...




## simple examples

so how does torch autograd work in practice?
let's see in a simple example for two tensors, $a = 4, b=3$.

```python
#simple grads:
a = torch.tensor(4, dtype=torch.float32, requires_grad=True)
b = torch.tensor(3, dtype=torch.float32, requires_grad=True)

c = a * b

#compute gradients for leaf nodes a, b
c.backward()
```

what do we expect to see when computing $\frac{\partial c}{\partial a}$? by simple calculus rules we expect $\frac{\partial c}{\partial a} = b = 3$. let's see the output:

```bash
a.grad: 3.0
b.grad: 4.0
```
nice!

note that the gradients are accumulated.
```python
c = a * b

#compute gradients for leaf nodes a, b
c.backward()

print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")

c = a * b

c.backward()
print(f"a.grad after second backward: {a.grad}")  # prints 6.0
```

this is useful when doing gradient accumulation when you want to simulate larger batch sizes than your hardware allows.


## neural network example
in order to generalize the example to neural networks, consider the following example. here we artificially construct a dataset with two inputs $x_1, x_2$ and a scalar output $y=x_1 + 0.1x_2^2$.

we want to learn a simple linear layer $\vec w = w_1, w_2$ to learn the data.


```python
dps = [     # generates the dataset
    (
        torch.tensor((x1, x2), dtype=torch.float32),
        torch.tensor(x1 + .1*x2**2)
    )
    for x1 in range(num_samples) for x2 in range(num_samples)
]       #len(dps) = num_samples^2

w = torch.tensor([0.0213, 0.00124], dtype=torch.float32, requires_grad=True)
x = dps[10][0]; y = dps[10][1]
y_hat = w @ x

loss = (y_hat - y)**2   #simple least squares
loss.backward()
```

the output shows: `w.grad: tensor([-57.8978, -34.7387])` for `x=tensor([5., 3.])`, `y=5.9`, with a loss of `33.52`.

now we know the loss and can calculate how to update the weights to reduce it.

so theoretically, what do we expect? the following is the analytical solution for the derivative of $w_1$:
\[
\frac{\partial L}{\partial w_1} = \frac{\partial (w_1 x_1 + w_2 x_2 - y)^2}{\partial w_1} = 2 (w_1 x_1 + w_2 x_2 - y) x_1
\\
\Rightarrow 2(0.0213 \cdot 5 + 0.00124\cdot 3 - 5.9) \cdot 5 = -57.89
\]
as we can see, our calculated gradient is correct!


what happens under the hood?
the autograd engine builds a computational graph dynamically, where each tensor operation creates a new node in the graph. each node keeps track of the operation that created it and its parent nodes. when we call `backward()`, pytorch traverses this graph in reverse order, applying the chain rule to compute gradients for each tensor that has `requires_grad=True`.


using gradient descent optimization, we can update the weight and rerun and see how it works:
```python
with torch.no_grad():
    w = w - lr * w.grad

# recalculate:
x = dps[53][0]; y = dps[53][1]
y_hat = w @ x
loss = (y_hat - y)**2
print(f"after one gradient step: input {x}, y: {y}, y_hat: {y_hat}, loss: {loss.item()}")
```

as you can see below, this significantly improves our prediction!
```bash
input tensor([5., 3.]), y: 5.9, y_hat: 0.11022000014781952, loss: 33.52155303955078
after one gradient step: input tensor([5., 3.]), y: 5.9, y_hat: 2.89, loss: 9.06
```


of course, we don't want the model to overfit and *only learn to fit* one sample. for this above case, we update the weights only based on one sample. to test the loss for the whole dataset even if only one sample is optimized, the following plot shows:



```python
# len(dps) = 64
sample = 53
x = dps[sample][0]; y = dps[sample][1]

for epoch in range(epochs):
    avg_loss = evaluate_over_dataset(dps, w)

    y_hat = w @ x
    loss = (y_hat - y)**2   #simple least squares
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

    print(f"epochs {epoch}, avg_loss: {avg_loss}, individual_loss: {loss.item()}")


```


the plot below shows a cool thing:

<figure>
    <img src="/posts/res/2025-11-13-13-27-49.png" width=300 >
</figure>

we are not overfitting, but really decreasing the overall loss. one sample carries enough signal to reduce overall loss, but only so much—once it's fit, there's no more information to extract from it, and the overall loss stagnates.
here the avg loss is around 26, while the individual loss is at 2.6.


in the next step the gradient is accumulated for the whole dataset and then updated using standard gradient descent.

```python
def forward_step(x, y, w):
    y_hat = w @ x
    loss = (y_hat - y)**2   #simple least squares
    return loss

def evaluate_over_dataset(dps, w):
    total_loss = 0
    for x, y in dps:
        total_loss += forward_step(x, y, w).item()
    return total_loss / len(dps)

def train_over_dataset(dps, w):
    for x, y in dps:
        loss = forward_step(x, y, w)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

for epoch in range(epochs):
    avg_loss = evaluate_over_dataset(dps, w)
    train_over_dataset(dps, w)

```

here all the operations are accumulated to a large computational graph. calling the final `loss.backward()`, the gradient is calculated for all operations. this converges to a lower avg loss of around 5 after 25 epochs.
<figure>
    <img src="/posts/res/2025-11-25-11-05-52.png" width=300 >
</figure>

of course we are not able to completely fit the data with only two weights, as the function includes a quadratic term.


this really simple network allows for better understanding of the autograd mechanism.
for only one sample, the computation graph looks like this:

<figure>
    <img src="/posts/res/2025-11-25-11-42-47.png" width=250 >
</figure>

so let's break it down.
the following code is used to collect the gradients for the graph, where the graph nodes are added directly to the code:

```python
def forward_step(x, y, w):
    y_hat = w @ x       # DotBackward0
    loss = (y_hat - y)  # SubBackward0
    loss = loss**2      # PowBackward0
    return loss

def evaluate_over_dataset(dps, w):
    total_loss = torch.tensor(0.0)
    for x, y in dps:
        total_loss += forward_step(x, y, w)   # AddBackward0
    return total_loss / len(dps) # DivBackward0
```

this is pretty simple! note that `PowBackward0` has an exponent of 2 and `DotBackward0` operates on $w \cdot x$ because of the chain rule. this is formalized in the equation above.


what happens if we do this with 4 samples instead of just one? we expect to have the same operations, but the respective samples are added before division.


<figure>
    <img src="/posts/res/2025-11-25-11-49-45.png" width=300 >
</figure>

and that's exactly what's happening!

so the computational graph for each sample looks the same, but before performing the avg calculation of the loss, all individual losses are summed up using `AddBackward` nodes.

now with even more samples! (n=36)

<figure>
    <img src="/posts/res/2025-11-25-11-51-24.png" width=300 >
</figure>

but this gets very hard to look at. however, the idea is really simple: pytorch builds a computational graph for all operations done in the neural network. when calling `loss.backward()`, the graph is traversed in reverse order and the chain rule is applied to calculate the gradients.



this visualization shows how gradient descent works in the loss landscape:

<figure>
    <img src="/posts/res/2025-11-13-13-45-57.png" width=300 >
</figure>


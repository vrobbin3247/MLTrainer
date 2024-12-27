from mlp import Draw

log = []
def learn(steps, learning_rate,xs,ys,n):
    for k in range(steps):

        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass
        loss.backward()

        # update
        for p in n.parameters():
            p.data += learning_rate * p.grad  # 0.1 is learning rate

        log.append((k, loss.data))
    for i in range(len(ypred)):
        print(i, ypred[i])
        Draw.draw_dot(ypred[i]).render(directory='doctest-output',filename=f"{i}").replace('\\', '/')

    return log









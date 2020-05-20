def train(model, train_loader, optimizer, loss_function, device="cpu", epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            batch_loss = loss_function(predictions, target)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            if (batch_idx+1) % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx+1}, Average Loss: {running_loss/(batch_idx + 1)}")
train(model, train_loader, optimizer, loss_function, device=device)
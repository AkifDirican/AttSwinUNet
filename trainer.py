import torch
import numpy as np
import copy 
from sklearn.metrics import f1_score
from tqdm import tqdm


def trainer(config,Net,train_loader,test_loader,optimizer,criteria,args):
    best_F1_score = 0.0
    num_epochs = int(config['epochs'])
    for ep in range(num_epochs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Net.train()
        epoch_loss = 0
        num_batches = len(train_loader)
        for itter, batch in enumerate(train_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = Net(img)
            loss          = criteria(msk_pred, msk) 
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()  
            percent = 100.0 * (itter + 1) / num_batches
            print(f"Epoch {ep+1}/{num_epochs} - Batch {itter+1}/{num_batches} ({percent:.1f}%) - Loss: {loss.item():.4f}")
        if itter%int(float(config['progress_p']) * len(train_loader))==0:
            print(f' Epoch>> {ep+1} and itteration {itter+1} Loss>> {((epoch_loss/(itter+1)))}')

        if (ep+1)%args.eval_interval==0:
            with torch.no_grad():
                print('val_mode')
                predictions, gt = [], []
                Net.eval()
                for batch in test_loader:
                    img = batch['image'].to(device, dtype=torch.float32)
                    msk = batch['mask'].float()
                    msk_pred = torch.sigmoid(Net(img))           # <-- convert logits → prob

                    gt.append((msk[0, 0] >= 0.5).cpu().numpy().astype(np.uint8))
                    pred_bin = (msk_pred[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)
                    predictions.append(pred_bin)

            predictions = np.asarray(predictions).ravel()
            gt = np.asarray(gt).ravel()

            #F1 score
            F1_score = f1_score(gt, predictions, zero_division=1)
            print(f'F1 / DSC : {F1_score:.4f}')
            if (F1_score) > best_F1_score:
                print('New best loss, saving...')
                best_F1_score = copy.deepcopy(F1_score)
                state = copy.deepcopy({'model_weights': Net.state_dict(), 'test_F1_score': F1_score})
                torch.save(state, args.saved_model)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, TensorDataset
from joco.utils.get_random_projection import get_random_projection


def update_model(model, mll, learning_rte, train_x, train_y, n_epochs, train_bsz=64):
    model = model.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte
    )
    train_bsz = min(len(train_y), train_bsz)
    train_dataset = TensorDataset(train_x.to(device), train_y.to(device))
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            scores = scores.to(device)
            output = model(inputs)
            loss = -mll(output, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model


def update_surrogate_models_joco(
    models_list,
    mlls_list,
    learning_rte,
    train_x,
    train_y,
    train_s,
    n_epochs,
    seed=0,
    train_bsz=64,
    update_jointly=True,
    use_rand_proj_instead=False,
):
    if use_rand_proj_instead:
        # get_random_projection 
        compressed_x_dim = models_list[-1].feature_extractor.output_dim # all x models have same compression dim, pick any one
        compressed_y_dim = len(models_list) - 1 # first model maps from y to s, others map from x to each dim of compressed y
        train_x = get_random_projection(train_x, target_dim=compressed_x_dim) 
        train_y = get_random_projection(train_y, target_dim=compressed_y_dim) 
        assert update_jointly 
    for model in models_list:
        model.train()
    torch_models_list = torch.nn.ModuleList(models_list)
    params_list = [{"params": torch_models_list.parameters(), "lr": learning_rte}]
    optimizer = torch.optim.Adam(params_list, lr=learning_rte)
    train_bsz = min(len(train_y), train_bsz)
    train_dataset = TensorDataset(
        train_x.to(device), train_y.to(device), train_s.to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    # ablate joco without joint model updates
    if not update_jointly:
        models_list = update_surrogate_models_joco_non_joint(
            train_loader=train_loader,
            learning_rte=learning_rte,
            models_list=models_list,
            mlls_list=mlls_list,
            n_epochs=n_epochs,
        )
    else:
        models_list = update_surrogate_models_joco_loss_v1(
            train_loader=train_loader,
            optimizer=optimizer,
            models_list=models_list,
            mlls_list=mlls_list,
            n_epochs=n_epochs,
            use_rand_proj_instead=use_rand_proj_instead,
        )

    for model in models_list:
        model.eval()

    return models_list


def update_gpx_models_only(
    gpy_model,
    train_loader,
    x_models_list,
    learning_rte,
    x_mlls_list,
    n_epochs,
):
    gpy_model = gpy_model.eval()
    n_x_models = len(x_models_list)
    # define seperate optimizers for each loss
    x_models_list_torch = torch.nn.ModuleList(x_models_list)
    x_params_list = [{"params": x_models_list_torch.parameters(), "lr": learning_rte}]
    x_optimizer = torch.optim.Adam(x_params_list, lr=learning_rte)
    for _ in range(n_epochs):
        for (xs, ys, _) in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            compressed_y = gpy_model.feature_extractor(ys)
            compressed_y = compressed_y.to(device)
            x_model_loss = 0.0
            for ix, model in enumerate(x_models_list):
                y_hat_dist = model(xs)
                loss = -x_mlls_list[ix](y_hat_dist, compressed_y[:, ix])
                loss = loss / n_x_models
                x_model_loss = x_model_loss + loss
            x_optimizer.zero_grad()  #
            x_model_loss.backward()  #
            x_optimizer.step()  #
            if device == "cuda":
                torch.cuda.empty_cache()

    return x_models_list


def update_gpy_model_only(
    gpy_model,
    gpy_mll,
    learning_rte,
    n_epochs,
    train_loader,
):
    gpy_model = gpy_model.train()
    y_params_list = [{"params": gpy_model.parameters(), "lr": learning_rte}]
    y_optimizer = torch.optim.Adam(y_params_list, lr=learning_rte)
    for _ in range(n_epochs):
        for (_, ys, rs) in train_loader:
            ys = ys.to(device)
            rs = rs.to(device)
            pred_r = gpy_model(ys)
            y_model_loss = -gpy_mll(pred_r, rs)
            y_optimizer.zero_grad()  #
            y_model_loss.backward()  #
            y_optimizer.step()  #
            if device == "cuda":
                torch.cuda.empty_cache()

    return gpy_model


def update_surrogate_models_joco_non_joint(
    train_loader,
    learning_rte,
    models_list,
    mlls_list,
    n_epochs,
):
    gp_x_models_list = update_gpx_models_only(
        gpy_model=models_list[0],
        train_loader=train_loader,
        x_models_list=models_list[1:],
        learning_rte=learning_rte,
        x_mlls_list=mlls_list[1:],
        n_epochs=n_epochs,
    )
    gp_y_model = update_gpy_model_only(
        gpy_model=models_list[0],
        gpy_mll=mlls_list[0],
        learning_rte=learning_rte,
        n_epochs=n_epochs,
        train_loader=train_loader,
    )

    models_list = [gp_y_model] + gp_x_models_list

    return models_list


def update_surrogate_models_joco_loss_v1(
    train_loader,
    optimizer,
    models_list,
    mlls_list,
    n_epochs,
    use_rand_proj_instead,
):
    # Assume first model is g(y) model
    first_model = models_list[0]
    x_models_list = models_list[1:]
    n_x_models = len(x_models_list)
    for _ in range(n_epochs):
        for (xs, ys, rs) in train_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            rs = rs.to(device)
            if use_rand_proj_instead:
                compressed_y = ys 
            else:
                compressed_y = first_model.feature_extractor(ys)
                compressed_y = compressed_y.to(device)
            # get g(y) model prediction and loss
            if use_rand_proj_instead:
                # ys already compressed by random projection, don't use dkl feature extractor 
                pred_r = first_model(ys, use_feature_extractor=False)
            else:
                pred_r = first_model(ys) # calls with dkl compression model 
            total_loss = -mlls_list[0](pred_r, rs)
            for ix, model in enumerate(x_models_list):
                # then get loss for each x model in trying to predict compressed version of y
                if use_rand_proj_instead:
                    # xs already compressed by random projection, don't use dkl feature extractor 
                    y_hat_dist = model(xs, use_feature_extractor=False)
                else:
                    y_hat_dist = model(xs) # calls with dkl compression model/ feautre extractor to compress 
                loss = -mlls_list[ix + 1](y_hat_dist, compressed_y[:, ix])
                loss = loss / n_x_models
                total_loss = total_loss + loss
            optimizer.zero_grad()  #
            total_loss.backward()  #
            optimizer.step()  #
            if device == "cuda":
                torch.cuda.empty_cache()

    return models_list

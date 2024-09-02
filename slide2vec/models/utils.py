def update_state_dict(model_dict, state_dict):
    success, shape_mismatch, missing_keys = 0, 0, 0
    updated_state_dict = {}
    shape_mismatch_list = []
    missing_keys_list = []
    for k, v in state_dict.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                updated_state_dict[k] = v
                success += 1
            else:
                updated_state_dict[k] = model_dict[k]
                shape_mismatch += 1
                shape_mismatch_list.append(k)
        else:
            missing_keys += 1
            missing_keys_list.append(k)
    if shape_mismatch > 0 or missing_keys > 0:
        msg = (f"{success}/{len(state_dict)} weight(s) loaded successfully\n"
           f"{shape_mismatch} weight(s) not loaded due to mismatching shapes: {shape_mismatch_list}\n"
           f"{missing_keys} key(s) not found in model: {missing_keys_list}")
    else:
        msg = f"{success}/{len(state_dict)} weight(s) loaded successfully."
    return updated_state_dict, msg
def my_sqrt(sample):
    return {"r": sample["id"] ** 0.5}


def return_id(sample):
    return {"r": sample["id"]}


global_var = 42


def test_can_init_global(sample):
    global global_var
    return {"r": global_var}

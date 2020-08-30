@register("fmnist_vae10_rd_mnist")
def fmnist_vae10_rd_mnist():
    Hparam = fmnist_vae10_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "fmnist_vae10_rd"
    return Hparam


@register("fmnist_vae1_rd_mnist")
def fmnist_vae1_rd_mnist():
    Hparam = fmnist_vae2_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 1
    Hparam.load_hparam_name = "fmnist_vae2_rd"
    return Hparam


@register("fmnist_vae2_rd_mnist")
def fmnist_vae2_rd_mnist():
    Hparam = fmnist_vae2_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "fmnist_vae2_rd"
    return Hparam


@register("fmnist_vae100_mnist")
def fmnist_vae100_mnist():
    Hparam = fmnist_vae100()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 100
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 50000
    Hparam.load_hparam_name = "fmnist_vae100"
    return Hparam


@register("mnist_vae10_rd_fmnist")
def mnist_vae10_rd_fmnist():
    Hparam = mnist_vae10_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "mnist_vae10_rd"
    return Hparam


@register("mnist_vae1_rd_fmnist")
def mnist_vae1_rd_fmnist():
    Hparam = mnist_vae1_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 1
    Hparam.load_hparam_name = "mnist_vae2_rd"
    return Hparam


@register("mnist_vae2_rd_fmnist")
def mnist_vae2_rd_fmnist():
    Hparam = mnist_vae2_rd()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "mnist_vae2_rd"
    return Hparam


@register("mnist_vae100_fmnist")
def mnist_vae100_fmnist():
    Hparam = mnist_vae100()
    Hparam.train_first = False
    # Hparam.model_train.z_size = 100
    # Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 50000
    Hparam.load_hparam_name = "mnist_vae100"
    return Hparam


@register("fmnist_linear_vae100_mnist")
def fmnist_linear_vae100_mnist():
    Hparam = fmnist_linear_vae100()
    # Hparam.train_first = False
    # Hparam.double_precision = True
    # Hparam.model_train.z_size = 100
    # Hparam.analytic_rd_curve = True
    # Hparam.analytical_elbo = True
    # Hparam.rd.num_betas = 50
    # Hparam.rd.max_beta = 200
    # Hparam.model_train.epochs = 100
    # Hparam.rd.anneal_steps = 5000
    # Hparam.svd = True
    # Hparam.rd.target_dist = "joint_xz"
    # Hparam.model_name = "vae_linear_fixed_var"
    # Hparam.distortion_limit = None
    return Hparam


@register("mnist_linear_vae100_fmnist")
def mnist_linear_vae100_fmnist():
    Hparam = mnist_linear_vae100()
    Hparam.train_first = False
    # Hparam.double_precision = True
    # Hparam.model_train.z_size = 100
    # Hparam.analytic_rd_curve = True
    # Hparam.analytical_elbo = True
    # Hparam.rd.num_betas = 50
    # Hparam.rd.max_beta = 200
    # Hparam.model_train.epochs = 100
    # Hparam.rd.anneal_steps = 5000
    # Hparam.svd = True
    # Hparam.rd.target_dist = "joint_xz"
    # Hparam.model_name = "vae_linear_fixed_var"
    # Hparam.distortion_limit = None
    return Hparam

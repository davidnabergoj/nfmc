import torch


def nuts(n_dim, potential, n_iterations: int = 1000, warmup_steps: int = 50):
    from pyro.infer.mcmc import NUTS, MCMC
    n_chains: int = 1

    def potential_wrapper(x_dict):
        x = torch.column_stack([x_dict[f'd{i}'] for i in range(len(x_dict))])
        return potential(x)

    initial_params = {f'd{i}': torch.randn(size=(n_chains,)) for i in range(n_dim)}
    kernel = NUTS(potential_fn=potential_wrapper)
    mcmc = MCMC(kernel, num_samples=n_iterations, warmup_steps=warmup_steps, initial_params=initial_params,
                num_chains=n_chains)
    mcmc.run()

    return torch.concat([mcmc.get_samples()[f'd{i}'][:, :, None] for i in range(n_dim)], dim=2)


if __name__ == '__main__':
    out = nuts(10, lambda x: torch.sum(x ** 2, dim=1))
    print(f'{out.shape = }')

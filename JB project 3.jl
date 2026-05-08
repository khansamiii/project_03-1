### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8d4769c7-f441-45f0-9b9b-54fb2f26e65b
begin
	using DifferentialEquations
	using LinearAlgebra
	using StaticArrays
	using Plots
end

# ╔═╡ f17103ea-06bf-11f1-a2b0-79e68ed152eb
md"""# Project_03 - Multibody Dynamic modeling

![Sliding compound pendulum with a support block connected to a spring and rotating compound pendulum](https://raw.githubusercontent.com/cooperrc/me5180-project_02/refs/heads/main/spring_compound-2_bodies.png)

In this project, a rigid bar is connected to a sliding block along a
horizontal tracks. The sliding block is connected to a spring that stretches and compresses. The rigid bar $L = 0.4~m$ acts as a compound pendulum.  

1. $x_1-y_1-$ describes block 1 position and orientation, $\theta_1$
2. $x_2-y_2-$ describes the rigid bar position and orientation, $\theta_2$

The applied forces are, 

1. Spring attached to block 1, $F = -k x_1$ where $k = 10~N/m$
2. gravity acting on block 1 and the rigid bar, $F_1 = -m_1g\hat{j}$ and $F_2 = -m_2 g\hat{j}$ where $m_1 = 0.1~kg$ and $m_2 = 0.3~kg$
 
In this project, you need to 

1. determine constraint equations $C(\mathbf{q},~t)$
2. Create an augmented solution method for the dynamic motion of these two moving parts
3. visualize the motion of the system as the two parts complete at least one oscillation
4. calculate and show (graph or vectors) the constraint forces acting on the 2-body system
"""

# ╔═╡ 198f5512-29c1-4a92-b235-2c4f628ee3f4
md" # Part 1 — Constraint Equations
Let $\theta_2=0$ correspond to the pendulum being in line with the positive x axis

There are two constraints on the motion of the block, namely

$$y_1=0$$
$$\theta_1=0$$

There are also two constraints due to the pin connection. These are

$$x_2=x_1+\frac{L}{2}\cos(\theta_2)$$
$$y_2=y_1+\frac{L}{2}\sin(\theta_2)$$

Therefore, we can write the constraint equation $C(q,t)$ as

$$C(q,t) = \begin{bmatrix}
y_1\\
\theta_1\\
x_1-x_2+\frac{L}{2}\cos(\theta_2)\\
y_1-y_2+\frac{L}{2}\sin(\theta_2)
\end{bmatrix}=0$$
"

# ╔═╡ 940ff481-da83-4bf5-8894-ea237df4a008
md" # Part 2 — Augmented Formulation
If we let the generalized coordinates be

$$q = \begin{bmatrix}
x_1\\
y_1\\
\theta_1\\
x_2\\
y_2\\
\theta_2\end{bmatrix}$$
then, we find $C_q$ to be the following

$$C_q = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0\\
1 & 0 & 0 & -1 & 0 & -\frac{L}{2}\sin(\theta_2)\\
0 & 1 & 0 & 0 & -1 & \frac{L}{2}\cos(\theta_2)
\end{bmatrix}$$

Additionally, we find that

$$Q_d = \begin{bmatrix}
0\\
0\\
\frac{L}{2}\dot{\theta_2}^2\cos(\theta_2)\\
\frac{L}{2}\dot{\theta_2}^2\sin(\theta_2)\\
\end{bmatrix}$$

Finally, both objects are affected by gravity, but only the block is affected by the spring force so,

$$Q_e = \begin{bmatrix}
-kx_1\\
-m_1g\\
0\\
0\\
-m_2g\\
0
\end{bmatrix}$$
"

# ╔═╡ f4529911-87c9-4827-83e8-281a1ba6fe47
# Parameters struct — holds all physical constants
struct PendulumParams
	m1::Float64
	m2::Float64
	L::Float64
	k::Float64
	g::Float64
	I2::Float64
	
	function PendulumParams(m1=0.1, m2=0.3, L=0.4, k=10.0, g=9.81)
		I2 = (1/12) * m2 * L^2
		new(m1, m2, L, k, g, I2)
	end
end

# ╔═╡ 0d9be664-d7c5-4084-add2-25e5418742d6
# Manual augmented dynamics — every matrix written out explicitly
function physics_system!(du, u, p, t)
	# Unpack state: q = u[1:6], dq = u[7:12]
	x1, y1, θ1, x2, y2, θ2 = u[1:6]
	dx1, dy1, dθ1, dx2, dy2, dθ2 = u[7:12]
	
	d = p.L / 2

	# Mass matrix M — diagonal, inertia of block rotation set to 1 (constrained)
	M = diagm([p.m1, p.m1, 1.0, p.m2, p.m2, p.I2])

	# Constraint Jacobian Cq — derived manually from C(q,t)
	Cq = [ 0.0  1.0  0.0   0.0   0.0   0.0        ;
		   0.0  0.0  1.0   0.0   0.0   0.0        ;
		   1.0  0.0  0.0  -1.0   0.0  -d*sin(θ2)  ;
		   0.0  1.0  0.0   0.0  -1.0   d*cos(θ2) ]

	# Generalized external forces Qe — spring on block, gravity on both
	Qe = [ -p.k * x1  ,
		   -p.m1 * p.g,
		    0.0        ,
		    0.0        ,
		   -p.m2 * p.g,
		    0.0        ]

	# Quadratic velocity term Qd — arises from differentiating Cq*dq
	Qd = [ 0.0                    ,
		   0.0                    ,
		   d * dθ2^2 * cos(θ2)   ,
		   d * dθ2^2 * sin(θ2)   ]

	# Augmented system: [M  Cq'; Cq  0] * [ddq; λ] = [Qe; Qd]
	LHS = [ M        Cq'        ;
		    Cq   zeros(4,4) ]

	RHS = [ Qe ; Qd ]

	sol_vector = LHS \ RHS

	ddq = sol_vector[1:6]   # accelerations
	λ   = sol_vector[7:10]  # Lagrange multipliers (constraint forces)

	# State derivative
	du[1:6]  = u[7:12]   # velocity
	du[7:12] = ddq        # acceleration
end

# ╔═╡ 3561521d-c5d0-4195-a776-cdc5ebc85b9f
begin
	p = PendulumParams()

	# Initial conditions — bar starts at -45 degrees
	θ2_init = -pi/4
	u0 = [0.0, 0.0, 0.0,
		  p.L/2 * cos(θ2_init), p.L/2 * sin(θ2_init), θ2_init,
		  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

	tspan = (0.0, 10.0)
	prob  = ODEProblem(physics_system!, u0, tspan, p)
	sol   = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

	t_steps = LinRange(0.0, 10.0, 300)
end

# ╔═╡ 9139c770-3d6d-44e2-93fb-ee8edc16fad6
# Part 3 — Time series plots
function plot_results(sol, t_steps)
	t    = collect(t_steps)
	x1   = [sol(ti)[1]  for ti in t]
	θ2   = [sol(ti)[6]  for ti in t]
	dx1  = [sol(ti)[7]  for ti in t]
	dθ2  = [sol(ti)[12] for ti in t]

	# Position plot
	p1 = plot(t, x1,
		label="Block x₁ (m)", lw=2, color=:steelblue,
		xlabel="Time (s)", ylabel="Position / Angle",
		title="Positions vs Time",
		legend=:topright, grid=true, gridalpha=0.3)
	plot!(p1, t, θ2,
		label="Bar θ₂ (rad)", lw=2, color=:crimson, linestyle=:dash)

	# Velocity plot
	p2 = plot(t, dx1,
		label="Block ẋ₁ (m/s)", lw=2, color=:steelblue, linestyle=:dash,
		xlabel="Time (s)", ylabel="Velocity",
		title="Velocities vs Time",
		legend=:topright, grid=true, gridalpha=0.3)
	plot!(p2, t, dθ2,
		label="Bar θ̇₂ (rad/s)", lw=2, color=:crimson, linestyle=:dash)

	# Phase portrait — block
	p3 = plot(x1, dx1,
		label="x₁ vs ẋ₁", lw=1.5, color=:steelblue,
		xlabel="x₁ (m)", ylabel="ẋ₁ (m/s)",
		title="Block Phase Portrait",
		legend=:topright, grid=true, gridalpha=0.3)
	scatter!(p3, [x1[1]], [dx1[1]], ms=7, color=:gold,
		marker=:star5, label="IC")

	# Phase portrait — bar
	p4 = plot(θ2, dθ2,
		label="θ₂ vs θ̇₂", lw=1.5, color=:crimson,
		xlabel="θ₂ (rad)", ylabel="θ̇₂ (rad/s)",
		title="Bar Phase Portrait",
		legend=:topright, grid=true, gridalpha=0.3)
	scatter!(p4, [θ2[1]], [dθ2[1]], ms=7, color=:gold,
		marker=:star5, label="IC")

	fig = plot(p1, p2, p3, p4,
		layout=(2,2), size=(1000, 700),
		left_margin=6Plots.mm, bottom_margin=5Plots.mm)
	display(fig)
	return fig
end

# ╔═╡ constraint_forces_cell
# Part 4 — Extract and plot constraint forces (Lagrange multipliers)
begin
	# Re-solve augmented system at each time step to extract λ
	function get_constraint_forces(sol, t_steps, p)
		R = zeros(length(t_steps), 4)
		for (i, ti) in enumerate(t_steps)
			u  = sol(ti)
			x1, y1, θ1, x2, y2, θ2 = u[1:6]
			dx1, dy1, dθ1, dx2, dy2, dθ2 = u[7:12]
			d = p.L / 2

			M  = diagm([p.m1, p.m1, 1.0, p.m2, p.m2, p.I2])
			Cq = [ 0.0  1.0  0.0   0.0   0.0   0.0       ;
				   0.0  0.0  1.0   0.0   0.0   0.0       ;
				   1.0  0.0  0.0  -1.0   0.0  -d*sin(θ2) ;
				   0.0  1.0  0.0   0.0  -1.0   d*cos(θ2)]
			Qe = [-p.k*x1, -p.m1*p.g, 0.0, 0.0, -p.m2*p.g, 0.0]
			Qd = [0.0, 0.0, d*dθ2^2*cos(θ2), d*dθ2^2*sin(θ2)]

			LHS = [M  Cq'; Cq  zeros(4,4)]
			RHS = [Qe; Qd]
			sv  = LHS \ RHS
			R[i, :] = sv[7:10]
		end
		return R
	end

	R = get_constraint_forces(sol, t_steps, p)

	# Plot constraint forces
	colors_cf = [:royalblue, :darkorange, :green3, :crimson]
	labels_cf = ["λ₁: y₁ normal (N)", "λ₂: θ₁ reaction (N·m)",
				 "λ₃: pin x (N)", "λ₄: pin y (N)"]

	p_cf = plot(collect(t_steps), R[:, 1],
		label=labels_cf[1], lw=2, color=colors_cf[1],
		xlabel="Time (s)", ylabel="Constraint Force / Moment",
		title="Lagrange Multipliers — Constraint Forces",
		legend=:topright, grid=true, gridalpha=0.3,
		size=(800, 380), left_margin=5Plots.mm, bottom_margin=5Plots.mm)
	for j in 2:4
		plot!(p_cf, collect(t_steps), R[:, j],
			label=labels_cf[j], lw=2, color=colors_cf[j])
	end
	hline!(p_cf, [0.0], color=:black, lw=1, alpha=0.4, label="")
	display(p_cf)
end

# ╔═╡ 09da7c42-730b-4325-b551-31338125a987
# Part 3 — Enhanced animation
function animate_pendulum(sol, p, R, t_steps)
	fps        = 30
	trail_len  = 20
	tip_x_hist = Float64[]
	tip_y_hist = Float64[]

	anim = @animate for (i, ti) in enumerate(t_steps)
		u   = sol(ti)
		x1  = u[1];  θ2 = u[6]

		pivot = [x1, 0.0]
		tip   = [x1 + p.L * cos(θ2),       p.L * sin(θ2)]
		com   = [x1 + (p.L/2) * cos(θ2),  (p.L/2) * sin(θ2)]

		push!(tip_x_hist, tip[1])
		push!(tip_y_hist, tip[2])
		if length(tip_x_hist) > trail_len
			popfirst!(tip_x_hist); popfirst!(tip_y_hist)
		end

		pl = plot(xlim=(-1.2, 1.2), ylim=(-0.65, 0.65),
			aspect_ratio=:equal, legend=false,
			background_color=:white, grid=false,
			title="t = $(round(ti, digits=2)) s  |  " *
				  "θ₂ = $(round(rad2deg(θ2), digits=1))°  |  " *
				  "x₁ = $(round(x1, digits=3)) m",
			titlefontsize=9)

		# Track rail
		plot!(pl, [-1.2, 1.2], [0.0, 0.0], lw=4, color=:gray30)
		plot!(pl, [-1.2, 1.2], [0.02, 0.02], lw=1, color=:gray60)

		# Wall anchor
		scatter!(pl, [-0.6], [0.0], marker=:square, ms=10, color=:gray20)

		# Spring (sinusoidal zigzag)
		n_coils  = 8
		x_spring = range(-0.6, x1, length=60)
		span     = x1 - (-0.6)
		y_spring = (abs(span) > 0.01 ?
			0.05 .* sin.(n_coils * π .* (x_spring .- (-0.6)) ./ span) :
			zeros(60))
		plot!(pl, collect(x_spring), y_spring, lw=2, color=:darkorange)

		# Trailing tip path
		if length(tip_x_hist) > 1
			n_tr = length(tip_x_hist)
			for j in 2:n_tr
				α = 0.1 + 0.6 * (j-1) / (n_tr-1)
				plot!(pl, tip_x_hist[j-1:j], tip_y_hist[j-1:j],
					color=:deepskyblue, lw=1.5, alpha=α)
			end
		end

		# Rigid bar
		plot!(pl, [pivot[1], tip[1]], [pivot[2], tip[2]],
			lw=6, color=:royalblue4)

		# Bar center of mass
		scatter!(pl, [com[1]], [com[2]], ms=6,
			color=:white, markerstrokecolor=:royalblue4,
			markerstrokewidth=2)

		# Tip
		scatter!(pl, [tip[1]], [tip[2]], ms=7,
			color=:deepskyblue, markerstrokecolor=:black,
			markerstrokewidth=1)

		# Pin joint
		scatter!(pl, [pivot[1]], [pivot[2]], ms=7,
			color=:red, markerstrokecolor=:black, markerstrokewidth=1)

		# Constraint force arrow at pin
		force_scale = 0.02
		if i <= size(R, 1) && norm(R[i, 3:4]) > 1e-6
			quiver!(pl, [pivot[1]], [pivot[2]],
				quiver=([force_scale * R[i, 3]],
						[force_scale * R[i, 4]]),
				color=:red, linewidth=2)
		end

		# Sliding block (rectangle)
		bw, bh = 0.07, 0.06
		plot!(pl,
			[x1-bw, x1+bw, x1+bw, x1-bw, x1-bw],
			[-bh/2, -bh/2,  bh/2,  bh/2, -bh/2],
			fill=true, fillcolor=:gray25, color=:black, lw=1.5)

		pl
	end

	gif(anim, "pendulum_sim_enhanced.gif", fps=fps)
end

# ╔═╡ 9ae78d0a-2427-4ee3-925d-0df7a27b238a
begin
	plot_results(sol, t_steps)
	savefig("simulation_plots.png")
	animate_pendulum(sol, p, R, t_steps)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
DifferentialEquations = "~7.17.0"
Plots = "~1.41.6"
StaticArrays = "~1.9.18"
"""

# ╔═╡ Cell order:
# ╟─f17103ea-06bf-11f1-a2b0-79e68ed152eb
# ╟─198f5512-29c1-4a92-b235-2c4f628ee3f4
# ╟─940ff481-da83-4bf5-8894-ea237df4a008
# ╠═8d4769c7-f441-45f0-9b9b-54fb2f26e65b
# ╠═f4529911-87c9-4827-83e8-281a1ba6fe47
# ╠═0d9be664-d7c5-4084-add2-25e5418742d6
# ╠═3561521d-c5d0-4195-a776-cdc5ebc85b9f
# ╠═9139c770-3d6d-44e2-93fb-ee8edc16fad6
# ╠═constraint_forces_cell
# ╠═09da7c42-730b-4325-b551-31338125a987
# ╠═9ae78d0a-2427-4ee3-925d-0df7a27b238a
# ╟─00000000-0000-0000-0000-000000000001

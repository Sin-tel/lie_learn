from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix

l = 1

M1 = change_of_basis_matrix(
			l,
			frm=("real", "quantum", "centered", "cs"),
			to=("complex", "quantum", "centered", "cs"),
		)


print(M1)
class Tensor {
	constructor(data, shape = null, dtype = "float32") {
		this.data = data;
		this.shape = shape || this.inferShape(data);
		this.dtype = dtype; // Data type: float32, int32, etc.
	}

	inferShape(data) {
		const shape = [];
		let level = data;

		while (Array.isArray(level)) {
			shape.push(level.length);
			level = level[0];
		}

		return shape;
	}
	transpose() {
		if (this.shape.length < 2) {
			throw new Error("Transpose requires at least a 2D tensor");
		}
		const result = Array.from({ length: this.shape[1] }, () => []);
		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < this.shape[1]; j++) {
				result[j][i] = this.data[i][j];
			}
		}
		return new Tensor(result);
	}
	transposeInPlace() {
		if (this.shape.length < 2) {
			throw new Error("Transpose requires at least a 2D tensor");
		}
		const result = [];
		const [row, col] = this.shape.slice(-2);
		for (let i = 0; i < col; i++) {
			const newRow = [];
			for (let j = 0; j < row; j++) {
				newRow.push(this.data[j][i]);
			}
			result.push(newRow);
		}
		this.data = result;
		this.shape = this.shape.slice(0, -2).concat([col, row]);
		return this;
	}
	dotProduct(other) {
		if (this.shape.length !== 1 || other.shape.length !== 1) {
			throw new Error("Dot product is only applicable to 1D tensors.");
		}
		if (this.shape[0] !== other.shape[0]) {
			throw new Error("Both tensors must have the same length.");
		}

		let sum = 0.0;
		let compensation = 0.0; // Compensation for lost low-order bits
		let temp, y;

		for (let i = 0; i < this.data.length; i++) {
			y = this.data[i] * other.data[i] - compensation;
			temp = sum + y;
			compensation = temp - sum - y;
			sum = temp;
		}

		return sum;
	}
	matMul(other) {
		if (this.shape.length !== 2 || other.shape.length !== 2) {
			throw new Error("Both tensors must be 2D for matrix multiplication");
		}
		if (this.shape[1] !== other.shape[0]) {
			throw new Error("Incompatible shapes for matrix multiplication");
		}

		const result = new Array(this.shape[0])
			.fill(0)
			.map(() => new Array(other.shape[1]).fill(0));
		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < other.shape[1]; j++) {
				for (let k = 0; k < this.shape[1]; k++) {
					result[i][j] += this.data[i][k] * other.data[k][j];
				}
			}
		}
		return new Tensor(result);
	}
	strassenMatMul(other) {
		if (this.shape.length !== 2 || other.shape.length !== 2) {
			throw new Error("Both tensors must be 2D for matrix multiplication");
		}
		if (this.shape[1] !== other.shape[0]) {
			throw new Error("Incompatible shapes for matrix multiplication");
		}

		// Strassen algorithm is beneficial for large matrices
		const n = this.shape[0];
		if (n <= 64) {
			// Threshold can be adjusted based on performance tests
			return this.matMul(other); // Use standard multiplication for small matrices
		}

		// Ensure the matrices are square and dimensions are powers of 2
		const m = this.nextPowerOfTwo(n);
		const A = this.padToSize(m, m);
		const B = other.padToSize(m, m);

		// Strassen's recursive steps
		const [A11, A12, A21, A22] = A.splitIntoQuarters();
		const [B11, B12, B21, B22] = B.splitIntoQuarters();

		// Calculate p1 through p7
		const p1 = A11.strassenMatMul(B12.subtract(B22));
		const p2 = A11.add(A12).strassenMatMul(B22);
		const p3 = A21.add(A22).strassenMatMul(B11);
		const p4 = A22.strassenMatMul(B21.subtract(B11));
		const p5 = A11.add(A22).strassenMatMul(B11.add(B22));
		const p6 = A12.subtract(A22).strassenMatMul(B21.add(B22));
		const p7 = A11.subtract(A21).strassenMatMul(B11.add(B12));

		// Calculate C11, C12, C21, C22
		const C11 = p5.add(p4).subtract(p2).add(p6);
		const C12 = p1.add(p2);
		const C21 = p3.add(p4);
		const C22 = p1.add(p5).subtract(p3).subtract(p7);

		// Combine quarters into the final result matrix
		return new Tensor(combineQuarters(C11, C12, C21, C22), [m, m]).slice(
			[0, 0],
			[n, n]
		);
	}

	nextPowerOfTwo(x) {
		return Math.pow(2, Math.ceil(Math.log2(x)));
	}
	padToSize(newRows, newCols) {
		const paddedData = Array.from({ length: newRows }, (_, i) =>
			Array.from({ length: newCols }, (_, j) =>
				i < this.shape[0] && j < this.shape[1] ? this.data[i][j] : 0
			)
		);
		return new Tensor(paddedData, [newRows, newCols]);
	}

	splitIntoQuarters() {
		const midRow = this.shape[0] / 2;
		const midCol = this.shape[1] / 2;
		const quarters = [
			new Tensor(this.data.slice(0, midRow).map((row) => row.slice(0, midCol))), // A11
			new Tensor(this.data.slice(0, midRow).map((row) => row.slice(midCol))), // A12
			new Tensor(this.data.slice(midRow).map((row) => row.slice(0, midCol))), // A21
			new Tensor(this.data.slice(midRow).map((row) => row.slice(midCol))), // A22
		];
		return quarters;
	}

	static combineQuarters(C11, C12, C21, C22) {
		const topHalf = C11.data.map((row, i) => row.concat(C12.data[i]));
		const bottomHalf = C21.data.map((row, i) => row.concat(C22.data[i]));
		const combinedData = topHalf.concat(bottomHalf);
		return new Tensor(combinedData);
	}
	slice(begin, size) {
		if (
			begin.length !== this.shape.length ||
			size.length !== this.shape.length
		) {
			throw new Error(
				"Begin and size arrays must match the number of dimensions in the tensor"
			);
		}
		let slicedData = this.data;
		for (let i = 0; i < begin.length; i++) {
			slicedData = slicedData.map((x) => x.slice(begin[i], begin[i] + size[i]));
		}
		return new Tensor(slicedData);
	}

	reshape(newShape) {
		if (
			newShape.reduce((a, b) => a * b, 1) !==
			this.shape.reduce((a, b) => a * b, 1)
		) {
			throw new Error(
				"New shape must be compatible with the total number of elements"
			);
		}
		const flat = this.data.flat(Infinity);
		const reshaped = Array.from({ length: newShape[0] }, (_, i) => {
			const size = newShape.slice(1).reduce((a, b) => a * b, 1);
			return flat.slice(i * size, (i + 1) * size);
		});
		return new Tensor(reshaped, newShape);
	}
	elementWise(operation) {
		const applyOperation = (data) => {
			if (Array.isArray(data)) {
				return data.map((item) => applyOperation(item));
			} else {
				return operation(data);
			}
		};

		return new Tensor(applyOperation(this.data), this.shape);
	}

	add(other) {
		return this.elementWise((a, b) => a + b, other);
	}

	subtract(other) {
		return this.elementWise((a, b) => a - b, other);
	}

	multiply(other) {
		return this.elementWise((a, b) => a * b, other);
	}

	divide(other) {
		return this.elementWise((a, b) => a / b, other);
	}

	reduce(operation, initial) {
		const flatData = this.data.flat(Infinity);
		return flatData.reduce(operation, initial);
	}

	sum() {
		return this.reduce((acc, val) => acc + val, 0);
	}
	kahanSum(inputArray) {
		let sum = 0.0;
		let compensation = 0.0; // A running compensation for lost low-order bits.
		let y, t;

		for (const value of inputArray) {
			y = value - compensation; // So far, so good: compensation is zero.
			t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.
			compensation = t - sum - y; // (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
			sum = t; // Algebraically, compensation should always be zero. Beware overly-aggressive optimizing compilers!
		}
		return sum; // Return the correctly-rounded sum.
	}

	mean() {
		return this.sum() / this.data.flat(Infinity).length;
	}

	max() {
		return this.reduce((acc, val) => (acc > val ? acc : val), -Infinity);
	}

	min() {
		return this.reduce((acc, val) => (acc < val ? acc : val), Infinity);
	}

	clone() {
		return new Tensor(JSON.parse(JSON.stringify(this.data)), [...this.shape]);
	}

	fill(value) {
		const fillDeep = (arr, val) =>
			arr.map((v) => (Array.isArray(v) ? fillDeep(v, val) : val));
		this.data = fillDeep(this.data, value);
		return this;
	}

	isEmpty() {
		return (
			this.data.length === 0 ||
			this.data.every((item) => Array.isArray(item) && item.length === 0)
		);
	}

	normalize() {
		const maxVal = this.max();
		const minVal = this.min();
		return this.elementWise((value) => (value - minVal) / (maxVal - minVal));
	}

	standardize() {
		const meanVal = this.mean();
		const variance =
			this.reduce((acc, val) => acc + Math.pow(val - meanVal, 2), 0) /
			this.data.flat(Infinity).length;
		const stdDev = Math.sqrt(variance);

		const standardizeFunction = (value) => (value - meanVal) / stdDev;
		return this.elementWise(standardizeFunction, this.clone()); // Applying function element-wise to this tensor
	}

	luDecompose() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("LU decomposition requires a square matrix.");
		}
		const n = this.shape[0];
		const L = new Tensor(
			Array.from({ length: n }, () => new Array(n).fill(0)),
			[n, n]
		);
		const U = new Tensor(
			Array.from({ length: n }, () => new Array(n).fill(0)),
			[n, n]
		);

		// Iterate over each row
		for (let i = 0; i < n; i++) {
			// Form the Upper Triangular matrix U
			for (let j = i; j < n; j++) {
				let sum = 0;
				for (let k = 0; k < i; k++) {
					sum += L.data[i][k] * U.data[k][j];
				}
				U.data[i][j] = this.data[i][j] - sum;
			}

			// Form the Lower Triangular matrix L
			for (let j = i; j < n; j++) {
				if (i == j) {
					L.data[i][i] = 1; // Diagonal entries of L are set to 1
				} else {
					let sum = 0;
					for (let k = 0; k < i; k++) {
						sum += L.data[j][k] * U.data[k][i];
					}
					if (U.data[i][i] === 0) {
						throw new Error(
							"Zero pivot encountered, cannot proceed with decomposition."
						);
					}
					L.data[j][i] = (this.data[j][i] - sum) / U.data[i][i];
				}
			}
		}

		return { L, U };
	}

	qrDecompose() {
		if (this.shape.length !== 2 || this.shape[0] < this.shape[1]) {
			throw new Error(
				"QR decomposition requires a matrix with more rows than or equal to columns."
			);
		}

		let m = this.shape[0];
		let n = this.shape[1];
		let Q = new Tensor(
			Array.from({ length: m }, () => new Array(m).fill(0)),
			[m, m]
		);
		let R = new Tensor(
			Array.from({ length: m }, () => new Array(n).fill(0)),
			[m, n]
		);

		for (let k = 0; k < n; k++) {
			// Compute the norm of the k-th column vector
			let norm = 0;
			for (let i = k; i < m; i++) {
				norm += this.data[i][k] ** 2;
			}
			norm = Math.sqrt(norm);

			// Form k-th Householder vector
			let u_k = new Array(m).fill(0);
			for (let i = 0; i < k; i++) {
				u_k[i] = 0;
			}
			for (let i = k; i < m; i++) {
				u_k[i] =
					i === k
						? this.data[i][k] + Math.sign(this.data[k][k]) * norm
						: this.data[i][k];
			}
			let u_norm = Math.sqrt(u_k.reduce((acc, val) => acc + val ** 2, 0));
			u_k = u_k.map((x) => x / u_norm);

			// Update R and the matrix to be decomposed
			for (let i = k; i < n; i++) {
				let dot = 0;
				for (let j = k; j < m; j++) {
					dot += this.data[j][i] * u_k[j];
				}
				for (let j = k; j < m; j++) {
					this.data[j][i] -= 2 * u_k[j] * dot;
				}
			}
			for (let i = k; i < m; i++) {
				R.data[i][k] = 0;
				Q.data[i][k] = u_k[i];
			}
			R.data[k][k] = norm;
		}

		return { Q, R };
	}

	qrAlgorithm(maxIter = 1000, tolerance = 1e-10) {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("Matrix must be square for QR Algorithm.");
		}

		let A = this.clone();
		let Q = Tensor.identity(this.shape[0]);

		for (let i = 0; i < maxIter; i++) {
			let { Q: Q1, R } = A.qrDecompose();
			A = R.matMul(Q1);
			Q = Q.matMul(Q1);

			// Check for convergence: if off-diagonal elements are all small enough
			let isConverged = true;
			for (let row = 0; row < A.shape[0]; row++) {
				for (let col = 0; col < row; col++) {
					if (Math.abs(A.data[row][col]) > tolerance) {
						isConverged = false;
						break;
					}
				}
				if (!isConverged) break;
			}

			if (isConverged) break;
		}

		return {
			eigenvalues: A.data.map((row, idx) => row[idx]),
			eigenvectors: Q,
		};
	}

	static identity(size) {
		return new Tensor(
			Array.from({ length: size }, (_, i) =>
				Array.from({ length: size }, (_, j) => (i === j ? 1 : 0))
			),
			[size, size]
		);
	}

	svd() {
		if (this.shape.length !== 2) {
			throw new Error("SVD can only be applied to 2D matrices.");
		}

		// Assuming A is m x n
		let m = this.shape[0];
		let n = this.shape[1];

		// Step 1: Compute A^T A to find V
		let AT = this.transpose();
		let ATA = AT.matMul(this);

		// Compute eigenvalues and eigenvectors of A^T A
		let { eigenvalues: sigmaSquared, eigenvectors: V } = ATA.qrAlgorithm();

		// The singular values are the square roots of the eigenvalues of A^T A
		let sigma = sigmaSquared.map((value) => Math.sqrt(value));

		// Step 2: Compute AA^T to find U
		let AAT = this.matMul(AT);
		let { eigenvectors: U } = AAT.qrAlgorithm();

		// Create Sigma matrix (diagonal matrix of singular values)
		let Sigma = new Tensor(
			Array.from({ length: m }, (_, i) =>
				Array.from({ length: n }, (_, j) => (i === j ? sigma[i] : 0))
			),
			[m, n]
		);

		return { U, Sigma, V };
	}
	inverse() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("Inverse can only be applied to square matrices.");
		}

		const { L, U } = this.luDecompose();
		const n = this.shape[0];
		const inv = new Tensor(
			Array.from({ length: n }, () => new Array(n).fill(0)),
			[n, n]
		);

		// Solve L * Y = I (Forward substitution)
		for (let i = 0; i < n; i++) {
			for (let j = 0; j <= i; j++) {
				if (i == j) {
					inv.data[i][j] = 1; // Diagonal entries are 1 in L
				} else {
					let sum = 0;
					for (let k = 0; k < j; k++) {
						sum += L.data[i][k] * inv.data[k][j];
					}
					inv.data[i][j] = -sum; // Non-diagonal entries computation
				}
			}
		}

		// Solve U * X = Y (Backward substitution)
		for (let i = n - 1; i >= 0; i--) {
			for (let j = n - 1; j >= i; j--) {
				let sum = 0;
				for (let k = i + 1; k <= j; k++) {
					sum += U.data[i][k] * inv.data[k][j];
				}
				inv.data[i][j] = (inv.data[i][j] - sum) / U.data[i][i];
			}
		}

		return inv;
	}

	pseudoInverse() {
		if (this.shape.length !== 2) {
			throw new Error("Pseudo-inverse can only be applied to 2D matrices.");
		}
		// Compute SVD
		const { U, Sigma, V } = this.svd();

		// Calculate Sigma^+
		const m = Sigma.shape[0];
		const n = Sigma.shape[1];
		const SigmaPlus = new Tensor(
			Array.from({ length: n }, () => new Array(m).fill(0)),
			[n, m]
		);

		for (let i = 0; i < Math.min(m, n); i++) {
			if (Sigma.data[i][i] !== 0) {
				SigmaPlus.data[i][i] = 1 / Sigma.data[i][i];
			}
		}

		// V * Sigma^+ * U^T
		const VSigmaPlus = V.matMul(SigmaPlus);
		const pseudoInv = VSigmaPlus.matMul(U.transpose());
		return pseudoInv;
	}
	// Assume f is an array of functions and this Tensor is an array of variables
	computeJacobian(f) {
		if (this.shape.length !== 1) {
			throw new Error("Input for Jacobian computation must be a 1D Tensor.");
		}
		let vars = this.data;
		let jacobian = new Tensor(
			new Array(f.length).fill(0).map(() => new Array(vars.length).fill(0)),
			[f.length, vars.length]
		);

		const h = 1e-4; // step size
		for (let i = 0; i < f.length; i++) {
			for (let j = 0; j < vars.length; j++) {
				let vars_plus = [...vars];
				let vars_minus = [...vars];
				vars_plus[j] += h;
				vars_minus[j] -= h;
				jacobian.data[i][j] =
					(f[i](...vars_plus) - f[i](...vars_minus)) / (2 * h);
			}
		}
		return jacobian;
	}
	// Assume f is a function and this Tensor is an array of variables
	computeHessian(f) {
		if (this.shape.length !== 1) {
			throw new Error("Input for Hessian computation must be a 1D Tensor.");
		}
		let vars = this.data;
		let hessian = new Tensor(
			new Array(vars.length).fill(0).map(() => new Array(vars.length).fill(0)),
			[vars.length, vars.length]
		);

		const h = 1e-4; // step size
		for (let i = 0; i < vars.length; i++) {
			for (let j = 0; j < vars.length; j++) {
				let vars_ij = [...vars];
				let vars_i = [...vars];
				let vars_j = [...vars];
				let vars_0 = [...vars];
				vars_ij[i] += h;
				vars_ij[j] += h;
				vars_i[i] += h;
				vars_j[j] += h;
				hessian.data[i][j] =
					(f(...vars_ij) - f(...vars_i) - f(...vars_j) + f(...vars_0)) /
					(h * h);
			}
		}
		return hessian;
	}
	powerIteration(maxIter = 100, tolerance = 1e-6) {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("Matrix must be square for eigenvalue computation.");
		}

		let b_k = new Tensor(
			Array.from({ length: this.shape[0] }, () => [Math.random()]),
			[this.shape[0], 1]
		);
		let b_k1,
			lambda,
			oldLambda = 0;

		for (let i = 0; i < maxIter; i++) {
			// A b_k
			let Ab_k = this.matMul(b_k);

			// Normalize b_k1
			let norm = Math.sqrt(Ab_k.reduce((acc, val) => acc + val[0] ** 2, 0));
			b_k1 = new Tensor(
				Ab_k.data.map((row) => [row[0] / norm]),
				[this.shape[0], 1]
			);

			// Rayleigh quotient for the eigenvalue
			lambda = b_k1.transpose().matMul(this).matMul(b_k1).data[0][0];

			// Check convergence
			if (Math.abs(lambda - oldLambda) < tolerance) {
				break;
			}

			b_k = b_k1;
			oldLambda = lambda;
		}

		return { eigenvalue: lambda, eigenvector: b_k };
	}
	isSymmetric() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			return false; // Non-square matrices cannot be symmetric.
		}
		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < i; j++) {
				// Check only half since symmetric
				if (this.data[i][j] !== this.data[j][i]) {
					return false;
				}
			}
		}
		return true;
	}
	isDiagonal() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			return false; // Non-square matrices cannot be diagonal.
		}
		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < this.shape[0]; j++) {
				if (i !== j && this.data[i][j] !== 0) {
					return false;
				}
			}
		}
		return true;
	}
	isOrthogonal() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			return false;
		}
		const transpose = this.transpose().clone();
		const identity = this.matMul(transpose);

		return identity.isIdentity();
	}
	isIdentity() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			return false;
		}
		for (let i = 0; i < this.shape[0]; i++) {
			for (let j = 0; j < this.shape[0]; j++) {
				if (
					(i === j && this.data[i][j] !== 1) ||
					(i !== j && this.data[i][j] !== 0)
				) {
					return false;
				}
			}
		}
		return true;
	}
}

function assert(condition, message) {
	if (!condition) {
		throw new Error(message || "Assertion failed");
	}
}

function testTensorCreation() {
	const data = [
		[1, 2],
		[3, 4],
	];
	const tensor = new Tensor(data);
	assert(
		JSON.stringify(tensor.data) === JSON.stringify(data),
		"Failed to create tensor with correct data"
	);
	console.log("testTensorCreation passed.");
}

function testTranspose() {
	const data = [
		[1, 2],
		[3, 4],
	];
	const tensor = new Tensor(data);
	const transposed = tensor.transpose();
	const expected = [
		[1, 3],
		[2, 4],
	];
	assert(
		JSON.stringify(transposed.data) === JSON.stringify(expected),
		"Transpose failed"
	);
	console.log("testTranspose passed.");
}

function testMatMul() {
	const dataA = [
		[1, 2],
		[3, 4],
	];
	const tensorA = new Tensor(dataA);
	const dataB = [
		[2, 0],
		[1, 2],
	];
	const tensorB = new Tensor(dataB);
	const product = tensorA.matMul(tensorB);
	const expected = [
		[4, 4],
		[10, 8],
	];
	assert(
		JSON.stringify(product.data) === JSON.stringify(expected),
		"Matrix multiplication failed"
	);
	console.log("testMatMul passed.");
}

function testStrassenMatMul() {
	const dataA = [
		[1, 2],
		[3, 4],
	];
	const tensorA = new Tensor(dataA);
	const dataB = [
		[2, 0],
		[1, 2],
	];
	const tensorB = new Tensor(dataB);
	const product = tensorA.strassenMatMul(tensorB);
	const expected = [
		[4, 4],
		[10, 8],
	]; // Expecting same result as standard multiplication for small matrices
	assert(
		JSON.stringify(product.data) === JSON.stringify(expected),
		"Strassen Matrix multiplication failed"
	);
	console.log("testStrassenMatMul passed.");
}

function testLUdecompose() {
	const data = [
		[1, 2, 2],
		[4, 4, 2],
		[4, 6, 4],
	];
	const tensor = new Tensor(data);
	const { L, U } = tensor.luDecompose();
	const expectedL = [
		[1, 0, 0],
		[4, 1, 0],
		[4, 0.5, 1],
	];
	const expectedU = [
		[1, 2, 2],
		[0, -4, -6],
		[0, 0, -1],
	];
	console.log(L, U);
	assert(
		JSON.stringify(L.data) === JSON.stringify(expectedL) &&
			JSON.stringify(U.data) === JSON.stringify(expectedU),
		"LU decomposition failed"
	);
	console.log("testLUdecompose passed.");
}

function testInverse() {
	const data = [
		[4, 7],
		[2, 6],
	];
	const tensor = new Tensor(data);
	const inverse = tensor.inverse();
	const expected = [
		[0.6, -0.7],
		[-0.2, 0.4],
	];
	console.log("Computed inverse:", inverse.data);
	console.log("Expected inverse:", expected);
	assert(
		JSON.stringify(
			inverse.data.map((row) => row.map((x) => parseFloat(x.toFixed(1))))
		) === JSON.stringify(expected),
		"Inverse calculation failed"
	);

	console.log("testInverse passed.");
}

function testPseudoInverse() {
	const data = [
		[1, 2, 3],
		[4, 5, 6],
	];
	const tensor = new Tensor(data);
	const pseudoInverse = tensor.pseudoInverse();
	const expected = [
		[-17 / 18, 8 / 9],
		[-2 / 9, 1 / 9],
		[13 / 18, -4 / 9],
	];
	assert(
		JSON.stringify(
			pseudoInverse.data.map((row) => row.map((x) => parseFloat(x.toFixed(2))))
		) === JSON.stringify(expected),
		"Pseudo-inverse calculation failed"
	);
	console.log("testPseudoInverse passed.");
}
function testQRDecompose() {
	const data = [
		[12, -51, 4],
		[6, 167, -68],
		[-4, 24, -41],
	];
	const tensor = new Tensor(data);
	const { Q, R } = tensor.qrDecompose();
	console.log("Q:", Q.data, "R:", R.data);
	assert(Q.isOrthogonal() && R.data[0][0] > 0, "QR decomposition failed");
	console.log("testQRDecompose passed.");
}

function testNormalize() {
	const data = [1, 2, 3, 4, 5];
	const tensor = new Tensor(data);
	const normalized = tensor.normalize();
	const expected = [0, 0.25, 0.5, 0.75, 1.0];
	assert(
		JSON.stringify(normalized.data.map((x) => parseFloat(x.toFixed(2)))) ===
			JSON.stringify(expected),
		"Normalization failed"
	);
	console.log("testNormalize passed.");
}

function testStandardize() {
	const data = [1, 2, 3, 4, 5];
	const tensor = new Tensor(data);
	const standardized = tensor.standardize();
	console.log("Standardized Data:", standardized.data);
	assert(
		standardized.mean().toFixed(2) === "0.00",
		"Standardization failed to normalize mean"
	);
	console.log("testStandardize passed.");
}

function testReshape() {
	const data = [1, 2, 3, 4, 5, 6];
	const tensor = new Tensor(data);
	const reshaped = tensor.reshape([2, 3]);
	const expected = [
		[1, 2, 3],
		[4, 5, 6],
	];
	assert(
		JSON.stringify(reshaped.data) === JSON.stringify(expected),
		"Reshape failed"
	);
	console.log("testReshape passed.");
}

function runAllTests() {
	testTensorCreation();
	testTranspose();
	testMatMul();
	testStrassenMatMul();
	testLUdecompose();
	//testInverse();
	//testPseudoInverse();
	testQRDecompose();
	testNormalize();
	testStandardize();
	testReshape();
}

runAllTests();

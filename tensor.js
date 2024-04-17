class Tensor {
	constructor(data, shape = null, dtype = "float64") {
		this.dtype = dtype;
		if (data instanceof Float32Array || data instanceof Float64Array) {
			this.data = data;
			this.shape = shape || [data.length];
			// Ensure dtype matches the type of the TypedArray
			this.dtype = data instanceof Float32Array ? "float32" : "float64";
		} else {
			// Automatically handle regular array conversion to TypedArray
			const conversion = this.convertToTypedArray(data, dtype);
			this.shape = shape || conversion.shape; // Use provided shape or inferred shape
			this.data = conversion.data;
		}
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
	convertToTypedArray(data, dtype = "float64") {
		const shape = this.inferShape(data);
		const flatData = this.flatten(data);
		switch (dtype) {
			case "float32":
				return { data: new Float32Array(flatData), shape };
			case "float64":
				return { data: new Float64Array(flatData), shape };
			default:
				throw new Error("Unsupported data type");
		}
	}

	flatten(data) {
		return data.reduce(
			(acc, val) => acc.concat(Array.isArray(val) ? this.flatten(val) : val),
			[]
		);
	}

	transpose() {
		if (this.shape.length !== 2) {
			throw new Error("Transpose requires a 2D tensor");
		}

		const rows = this.shape[0];
		const cols = this.shape[1];
		const result = new this.data.constructor(rows * cols); // Create a new TypedArray of the same type
		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < cols; j++) {
				result[j * rows + i] = this.data[i * cols + j];
			}
		}
		return new Tensor(result, [cols, rows], this.dtype);
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

		// Create a new TypedArray for the result.
		const rowsA = this.shape[0];
		const colsA = this.shape[1];
		const colsB = other.shape[1];
		const result = new this.data.constructor(rowsA * colsB);

		for (let i = 0; i < rowsA; i++) {
			for (let j = 0; j < colsB; j++) {
				let sum = 0;
				for (let k = 0; k < colsA; k++) {
					sum += this.data[i * colsA + k] * other.data[k * colsB + j];
				}
				result[i * colsB + j] = sum;
			}
		}
		return new Tensor(result, [rowsA, colsB], this.dtype);
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
		const paddedData = new this.data.constructor(newRows * newCols);
		const originalCols = this.shape[1];

		for (let i = 0; i < newRows; i++) {
			for (let j = 0; j < newCols; j++) {
				if (i < this.shape[0] && j < this.shape[1]) {
					paddedData[i * newCols + j] = this.data[i * originalCols + j];
				} else {
					paddedData[i * newCols + j] = 0;
				}
			}
		}
		return new Tensor(paddedData, [newRows, newCols], this.dtype);
	}

	splitIntoQuarters() {
		const midRow = this.shape[0] / 2;
		const midCol = this.shape[1] / 2;
		const rowLength = this.shape[1];

		const A11 = new Tensor(
			new this.data.constructor(midRow * midCol),
			[midRow, midCol],
			this.dtype
		);
		const A12 = new Tensor(
			new this.data.constructor(midRow * midCol),
			[midRow, midCol],
			this.dtype
		);
		const A21 = new Tensor(
			new this.data.constructor(midRow * midCol),
			[midRow, midCol],
			this.dtype
		);
		const A22 = new Tensor(
			new this.data.constructor(midRow * midCol),
			[midRow, midCol],
			this.dtype
		);

		for (let i = 0; i < midRow; i++) {
			for (let j = 0; j < midCol; j++) {
				A11.data[i * midCol + j] = this.data[i * rowLength + j];
				A12.data[i * midCol + j] = this.data[i * rowLength + j + midCol];
				A21.data[i * midCol + j] = this.data[(i + midRow) * rowLength + j];
				A22.data[i * midCol + j] =
					this.data[(i + midRow) * rowLength + j + midCol];
			}
		}

		return [A11, A12, A21, A22];
	}

	static combineQuarters(C11, C12, C21, C22) {
		const m = C11.shape[0] + C21.shape[0];
		const n = C11.shape[1] + C12.shape[1];
		const combinedData = new C11.data.constructor(m * n);
		const midRow = C11.shape[0];
		const midCol = C11.shape[1];

		for (let i = 0; i < m; i++) {
			for (let j = 0; j < n; j++) {
				if (i < midRow && j < midCol) {
					combinedData[i * n + j] = C11.data[i * midCol + j];
				} else if (i < midRow) {
					combinedData[i * n + j] = C12.data[i * midCol + (j - midCol)];
				} else if (j < midCol) {
					combinedData[i * n + j] = C21.data[(i - midRow) * midCol + j];
				} else {
					combinedData[i * n + j] =
						C22.data[(i - midRow) * midCol + (j - midCol)];
				}
			}
		}

		return new Tensor(combinedData, [m, n], C11.dtype);
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

		const resultShape = size;
		const totalElements = resultShape.reduce((a, b) => a * b, 1);
		const resultData = new this.data.constructor(totalElements);
		let offset = 0;

		for (let i = 0; i < this.shape[0]; i++) {
			if (i < begin[0] || i >= begin[0] + size[0]) continue;
			const rowOffset = i * this.shape[1];

			for (let j = 0; j < this.shape[1]; j++) {
				if (j < begin[1] || j >= begin[1] + size[1]) continue;
				resultData[offset] = this.data[rowOffset + j];
				offset++;
			}
		}

		return new Tensor(resultData, resultShape, this.dtype);
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

		const totalElements = newShape.reduce((a, b) => a * b, 1);
		if (totalElements !== this.data.length) {
			throw new Error(
				"Total number of elements must remain constant during reshape."
			);
		}

		// Directly use the existing data since TypedArray does not need to be altered structurally
		return new Tensor(this.data, newShape, this.dtype);
	}

	elementWise(operation, other) {
		if (
			this.shape.length !== other.shape.length ||
			!this.shape.every((size, index) => size === other.shape[index])
		) {
			throw new Error("Shape mismatch for element-wise operation.");
		}
		const result = new this.data.constructor(this.data.length);
		for (let i = 0; i < this.data.length; i++) {
			result[i] = operation(this.data[i], other.data[i]);
		}
		return new Tensor(result, this.shape, this.dtype);
	}

	add(other) {
		if (
			this.shape.length !== other.shape.length ||
			!this.shape.every((size, index) => size === other.shape[index])
		) {
			throw new Error("Shape mismatch for addition.");
		}
		const result = new this.data.constructor(this.data.length);
		for (let i = 0; i < this.data.length; i++) {
			result[i] = this.data[i] + other.data[i];
		}
		return new Tensor(result, this.shape, this.dtype);
	}

	subtract(other) {
		if (
			this.shape.length !== other.shape.length ||
			!this.shape.every((size, index) => size === other.shape[index])
		) {
			throw new Error("Shape mismatch for subtraction.");
		}
		const result = new this.data.constructor(this.data.length);
		for (let i = 0; i < this.data.length; i++) {
			result[i] = this.data[i] - other.data[i];
		}
		return new Tensor(result, this.shape, this.dtype);
	}

	multiply(other) {
		return this.elementWise((a, b) => a * b, other);
	}

	divide(other) {
		return this.elementWise((a, b) => a / b, other);
	}

	reduce(operation, initial) {
		return this.data.reduce(operation, initial);
	}

	sum() {
		return this.reduce((acc, val) => acc + val, 0);
	}
	kahanSum() {
		let sum = 0.0;
		let compensation = 0.0; // A running compensation for lost low-order bits.
		let y, t;

		for (const value of this.data) {
			y = value - compensation; // So far, so good: compensation is zero.
			t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.
			compensation = t - sum - y; // (t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
			sum = t; // Algebraically, compensation should always be zero. Beware overly-aggressive optimizing compilers!
		}
		return sum; // Return the correctly-rounded sum.
	}

	mean() {
		return this.sum() / this.data.length;
	}

	max() {
		return this.reduce((acc, val) => Math.max(acc, val), -Infinity);
	}

	min() {
		return this.reduce((acc, val) => Math.min(acc, val), Infinity);
	}

	clone() {
		// Copy the underlying TypedArray data correctly
		const clonedData = new this.data.constructor(this.data);
		return new Tensor(clonedData, [...this.shape], this.dtype);
	}

	fill(value) {
		this.data.fill(value);
		return this;
	}

	isEmpty() {
		return this.data.length === 0;
	}

	normalize() {
		const maxVal = this.max();
		const minVal = this.min();
		if (maxVal === minVal) {
			// All elements are the same; avoid division by zero
			this.data.fill(0); // or any other constant value deemed appropriate
			return this;
		}

		const range = maxVal - minVal;
		const data = new this.data.constructor(this.data.length);
		for (let i = 0; i < this.data.length; i++) {
			data[i] = (this.data[i] - minVal) / range;
		}
		return new Tensor(data, this.shape, this.dtype);
	}

	standardize() {
		const meanVal = this.mean();
		const totalElements = this.data.length;
		let variance = 0;

		for (let i = 0; i < totalElements; i++) {
			let diff = this.data[i] - meanVal;
			variance += diff * diff;
		}
		variance /= totalElements;
		const stdDev = Math.sqrt(variance);

		if (stdDev === 0) {
			return this.fill(0); // Handle case where all values are the same
		}

		const result = new this.data.constructor(totalElements);
		for (let i = 0; i < totalElements; i++) {
			result[i] = (this.data[i] - meanVal) / stdDev;
		}

		return new Tensor(result, this.shape, this.dtype);
	}

	luDecompose() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("LU decomposition requires a square matrix.");
		}
		const n = this.shape[0];
		const LData =
			this.dtype === "float32"
				? new Float32Array(n * n)
				: new Float64Array(n * n);
		const UData =
			this.dtype === "float32"
				? new Float32Array(n * n)
				: new Float64Array(n * n);
		const L = new Tensor(LData, [n, n], this.dtype);
		const U = new Tensor(UData, [n, n], this.dtype);

		// Initialize L as identity matrix and U as zero matrix
		for (let i = 0; i < n; i++) {
			L.data[i * n + i] = 1;
			for (let j = 0; j < n; j++) {
				U.data[i * n + j] = 0;
			}
		}

		for (let i = 0; i < n; i++) {
			// Calculate U
			for (let j = i; j < n; j++) {
				let sum = 0;
				for (let k = 0; k < i; k++) {
					sum += L.data[i * n + k] * U.data[k * n + j];
				}
				U.data[i * n + j] = this.data[i * n + j] - sum;
			}
			// Calculate L
			for (let j = i + 1; j < n; j++) {
				let sum = 0;
				for (let k = 0; k < i; k++) {
					sum += L.data[j * n + k] * U.data[k * n + i];
				}
				L.data[j * n + i] = (this.data[j * n + i] - sum) / U.data[i * n + i];
			}
		}

		return { L, U };
	}

	qrDecompose() {
		const m = this.shape[0];
		const n = this.shape[1];
		let Q = Tensor.identity(m, this.dtype);
		let R = new Tensor(new Float64Array(m * n), [m, n], this.dtype);

		for (let k = 0; k < n; k++) {
			let column = this.getColumnTensor(k);
			let norm = Math.sqrt(column.reduce((acc, val) => acc + val * val, 0));

			if (norm === 0) {
				// Handle zero norm to prevent division by zero
				console.warn(
					`Column ${k} has zero norm, which may lead to instability.`
				);
				continue; // Optionally handle this scenario more gracefully
			}

			R.set(k, k, norm);
			for (let i = 0; i < m; i++) {
				Q.set(i, k, column.data[i] / norm); // Normalize column of A for Q
			}

			for (let j = k + 1; j < n; j++) {
				let columnJ = this.getColumnTensor(j);
				let dotProduct = Q.getColumnTensor(k).dotProduct(columnJ);
				R.set(k, j, dotProduct); // R[k, j] = dot product

				// Adjust the j-th column of A
				for (let i = 0; i < m; i++) {
					this.data[i * n + j] -= Q.data[i * m + k] * dotProduct; // A[i, j] -= Q[i, k] * dotProduct
				}
			}
		}

		return { Q, R };
	}

	getColumnTensor(k) {
		const numRows = this.shape[0];
		const columnData = new Float64Array(numRows);
		for (let i = 0; i < numRows; i++) {
			columnData[i] = this.data[i * this.shape[1] + k];
		}
		return new Tensor(columnData, [numRows], this.dtype); // Ensure this is a 1D tensor
	}

	set(row, col, value) {
		if (row >= this.shape[0] || col >= this.shape[1]) {
			throw new Error("Index out of bounds");
		}
		this.data[row * this.shape[1] + col] = value;
	}

	qrAlgorithm(maxIter = 1000, tolerance = 1e-16) {
		let A = this.clone();
		let Q = Tensor.identity(this.shape[0], this.dtype);

		let previousOffDiagonalSum = Infinity;
		for (let iter = 0; iter < maxIter; iter++) {
			let { Q: Q1, R } = A.qrDecompose();
			A = R.matMul(Q1);
			Q = Q.matMul(Q1);

			let offDiagonalSum = 0;
			for (let row = 0; row < A.shape[0]; row++) {
				for (let col = 0; col < A.shape[1]; col++) {
					if (row !== col) {
						offDiagonalSum += Math.abs(A.data[row * A.shape[1] + col]);
					}
				}
			}

			// Debugging output
			//console.log(`Iteration ${iter}: Off-diagonal sum = ${offDiagonalSum}`);

			// Check for convergence
			if (Math.abs(offDiagonalSum - previousOffDiagonalSum) < tolerance) {
				console.log("Converged at iteration", iter);
				break;
			}
			previousOffDiagonalSum = offDiagonalSum;
		}

		let eigenvalues = new this.data.constructor(A.shape[0]);
		for (let i = 0; i < A.shape[0]; i++) {
			eigenvalues[i] = A.data[i * A.shape[1] + i];
		}
		//console.log("eigenvals:", eigenvalues, "eigenvects:", Q);
		return {
			eigenvalues: eigenvalues,
			eigenvectors: Q,
		};
	}

	static identity(size, dtype = "float64") {
		let data;
		if (dtype === "float32") {
			data = new Float32Array(size * size);
		} else {
			data = new Float64Array(size * size);
		}
		for (let i = 0; i < size; i++) {
			data[i * size + i] = 1;
		}
		return new Tensor(data, [size, size], dtype);
	}

	svd() {
		if (this.shape.length !== 2) {
			throw new Error("SVD can only be applied to 2D matrices.");
		}

		const m = this.shape[0];
		const n = this.shape[1];
		const AT = this.transpose();
		const ATA = AT.matMul(this);
		const { eigenvalues: sigmaSquared, eigenvectors: V } = ATA.qrAlgorithm();

		let sigma = new Float64Array(Math.min(m, n));
		for (let i = 0; i < sigma.length; i++) {
			sigma[i] = Math.sqrt(Math.max(0, sigmaSquared[i])); // Ensure non-negative
		}

		let AAT = this.matMul(AT);
		let { eigenvectors: U } = AAT.qrAlgorithm();

		let Sigma = new Tensor(new Float64Array(m * n), [m, n], this.dtype).fill(0);
		for (let i = 0; i < sigma.length; i++) {
			Sigma.data[i * Sigma.shape[1] + i] = sigma[i];
		}
		console.log("U matrix:", U.data);
		console.log("Sigma matrix:", Sigma.data);
		console.log("V matrix:", V.data);
		
		return { U, Sigma, V };
	}
	conditionNumber() {
		const svdResult = this.svd(); // Apply SVD to the current matrix
		const { Sigma } = svdResult;
		const singularValues = Array.from(Sigma.data).filter((val) => val !== 0); // Filter out zero values
		const max = Math.max(...singularValues);
		const min = Math.min(...singularValues);

		if (min === 0) return Infinity; // Return Infinity if any singular value is zero

		return max / min;
	}

	inverse() {
		if (this.shape.length !== 2 || this.shape[0] !== this.shape[1]) {
			throw new Error("Inverse can only be applied to square matrices.");
		}

		const n = this.shape[0];
		let A = this.clone(); // Clone to avoid modifying original data
		let inv = Tensor.identity(n, this.dtype);

		// Perform Gauss-Jordan Elimination
		for (let i = 0; i < n; i++) {
			// Find the row with the maximum element in the current column
			let maxRow = i;
			for (let j = i + 1; j < n; j++) {
				if (Math.abs(A.data[j * n + i]) > Math.abs(A.data[maxRow * n + i])) {
					maxRow = j;
				}
			}

			// Swap the maximum row with the current row
			if (i !== maxRow) {
				A.swapRows(i, maxRow);
				inv.swapRows(i, maxRow);
			}

			// Scale the row to make the diagonal element 1
			const factor = A.data[i * n + i];
			if (factor === 0) {
				throw new Error("Matrix is singular and cannot be inverted.");
			}
			for (let j = 0; j < n; j++) {
				A.data[i * n + j] /= factor;
				inv.data[i * n + j] /= factor;
			}

			// Eliminate all other entries in this column
			for (let j = 0; j < n; j++) {
				if (j !== i) {
					const factor = A.data[j * n + i];
					for (let k = 0; k < n; k++) {
						A.data[j * n + k] -= A.data[i * n + k] * factor;
						inv.data[j * n + k] -= inv.data[i * n + k] * factor;
					}
				}
			}
		}

		return inv;
	}
	pseudoInverse(regularizationThreshold = 1e-15) {
		if (this.shape.length !== 2) {
			throw new Error("Pseudo-inverse can only be applied to 2D matrices.");
		}

		// Compute the SVD of the matrix
		const { U, Sigma, V } = this.svd();
		const m = Sigma.shape[0];
		const n = Sigma.shape[1];
		let SigmaPlusData = new (this.dtype === "float32"
			? Float32Array
			: Float64Array)(n * m).fill(0);
		let SigmaPlus = new Tensor(SigmaPlusData, [n, m], this.dtype);

		// Find the maximum singular value to determine the cutoff for regularization
		let maxSigma = 0;
		for (let i = 0; i < Math.min(m, n); i++) {
			if (Sigma.data[i * n + i] > maxSigma) {
				maxSigma = Sigma.data[i * n + i];
			}
		}

		// Regularize singular values that are significantly smaller than the maximum singular value
		for (let i = 0; i < Math.min(m, n); i++) {
			const sigma = Sigma.data[i * n + i];
			if (sigma / maxSigma > regularizationThreshold) {
				SigmaPlus.data[i * m + i] = 1 / sigma;
			} else {
				SigmaPlus.data[i * m + i] = 0; // Treat very small singular values as zero
			}
		}

		// Compute the pseudo-inverse using the regularized Sigma+
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
		const numRows = f.length;
		const numCols = vars.length;
		const jacobianData =
			this.dtype === "float32"
				? new Float32Array(numRows * numCols)
				: new Float64Array(numRows * numCols);
		let jacobian = new Tensor(jacobianData, [numRows, numCols], this.dtype);
		const h = 1e-4; // step size
		for (let i = 0; i < numRows; i++) {
			for (let j = 0; j < numCols; j++) {
				let vars_plus = new this.data.constructor(vars); // Copy vars array
				let vars_minus = new this.data.constructor(vars); // Copy vars array
				vars_plus[j] += h;
				vars_minus[j] -= h;
				jacobian.data[i * numCols + j] =
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
		const numVars = vars.length;
		const hessianData =
			this.dtype === "float32"
				? new Float32Array(numVars * numVars)
				: new Float64Array(numVars * numVars);
		let hessian = new Tensor(hessianData, [numVars, numVars], this.dtype);

		const h = 1e-4; // step size
		for (let i = 0; i < numVars; i++) {
			for (let j = 0; j < numVars; j++) {
				let vars_ij = new this.data.constructor(vars); // Copy vars array
				let vars_i = new this.data.constructor(vars); // Copy vars array
				let vars_j = new this.data.constructor(vars); // Copy vars array
				let vars_0 = new this.data.constructor(vars); // Copy vars array
				vars_ij[i] += h;
				vars_ij[j] += h;
				vars_i[i] += h;
				vars_j[j] += h;
				hessian.data[i * numVars + j] =
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
		const size = this.shape[0];
		let b_kData =
			this.dtype === "float32"
				? new Float32Array(size).fill(Math.random())
				: new Float64Array(size).fill(Math.random());
		let b_k = new Tensor(b_kData, [size, 1], this.dtype);
		let b_k1,
			lambda,
			oldLambda = 0;

		for (let i = 0; i < maxIter; i++) {
			// A b_k
			let Ab_k = this.matMul(b_k);

			// Normalize b_k1
			let norm = Math.sqrt(Ab_k.data.reduce((acc, val) => acc + val ** 2, 0));
			let normalizedData = new this.data.constructor(size);
			for (let j = 0; j < size; j++) {
				normalizedData[j] = Ab_k.data[j] / norm;
			}
			b_k1 = new Tensor(normalizedData, [size, 1], this.dtype);

			// Rayleigh quotient for the eigenvalue
			lambda = b_k1.transpose().matMul(this).matMul(b_k1).data[0];

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
				if (
					this.data[i * this.shape[1] + j] !== this.data[j * this.shape[1] + i]
				) {
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
				if (i !== j && this.data[i * this.shape[1] + j] !== 0) {
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
		const transpose = this.transpose();
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
					(i === j && this.data[i * this.shape[1] + j] !== 1) ||
					(i !== j && this.data[i * this.shape[1] + j] !== 0)
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
function arraysEqual(a, b) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) return false;
	}
	return true;
}
function arraysAlmostEqual(a, b, tolerance = 1e-4) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (Math.abs(a[i] - b[i]) > tolerance) return false;
	}
	return true;
}
function testTensorCreation() {
	const data = [
		[1, 2],
		[3, 4],
	];
	const tensor = new Tensor(data);
	const flatData = new Float64Array([1, 2, 3, 4]); // Assuming dtype is float64
	assert(
		arraysEqual(tensor.data, flatData),
		"Failed to create tensor with correct data"
	);
	console.log("testTensorCreation passed.");
}
function testAdd() {
	const tensor1 = new Tensor([1, 2, 3]);
	const tensor2 = new Tensor([4, 5, 6]);
	const result = tensor1.add(tensor2);
	const expected = new Float64Array([5, 7, 9]);
	assert(arraysEqual(result.data, expected), "Addition failed");
	console.log("testAdd passed.");
}

function testSubtract() {
	const tensor1 = new Tensor([10, 20, 30]);
	const tensor2 = new Tensor([1, 2, 3]);
	const result = tensor1.subtract(tensor2);
	const expected = new Float64Array([9, 18, 27]);
	assert(arraysEqual(result.data, expected), "Subtraction failed");
	console.log("testSubtract passed.");
}

function testMultiply() {
	const tensor1 = new Tensor([1, 2, 3]);
	const tensor2 = new Tensor([4, 5, 6]);
	const result = tensor1.multiply(tensor2);
	const expected = new Float64Array([4, 10, 18]);
	assert(
		arraysEqual(result.data, expected),
		"Element-wise multiplication failed"
	);
	console.log("testMultiply passed.");
}

function testDivide() {
	const tensor1 = new Tensor([10, 20, 30]);
	const tensor2 = new Tensor([2, 4, 5]);
	const result = tensor1.divide(tensor2);
	const expected = new Float64Array([5, 5, 6]);
	assert(arraysEqual(result.data, expected), "Element-wise division failed");
	console.log("testDivide passed.");
}

function testMax() {
	const tensor = new Tensor([1, 3, 2, 8, 5]);
	const result = tensor.max();
	assert(result === 8, "Maximum value incorrect");
	console.log("testMax passed.");
}

function testMin() {
	const tensor = new Tensor([10, 3, 2, 8, 5]);
	const result = tensor.min();
	assert(result === 2, "Minimum value incorrect");
	console.log("testMin passed.");
}

function testSum() {
	const tensor = new Tensor([1, 2, 3, 4]);
	const result = tensor.sum();
	assert(result === 10, "Sum calculation failed");
	console.log("testSum passed.");
}

function testKahanSum() {
	const tensor = new Tensor([0.1, 0.2, 0.3]);
	const result = tensor.kahanSum();
	assert(Math.abs(result - 0.6) < 1e-15, "Kahan sum calculation failed");
	console.log("testKahanSum passed.");
}

function testIsEmpty() {
	const tensor = new Tensor([]);
	assert(
		tensor.isEmpty() === true,
		"IsEmpty should return true for empty tensor"
	);
	console.log("testIsEmpty passed.");
}

function testFill() {
	const tensor = new Tensor([1, 2, 3, 4, 5]);
	tensor.fill(0);
	const expected = new Float64Array([0, 0, 0, 0, 0]);
	assert(arraysEqual(tensor.data, expected), "Fill method failed");
	console.log("testFill passed.");
}

function testClone() {
	const tensor = new Tensor([1, 2, 3]);
	const clonedTensor = tensor.clone();
	assert(
		arraysEqual(tensor.data, clonedTensor.data) &&
			tensor.shape[0] === clonedTensor.shape[0],
		"Clone method failed"
	);
	console.log("testClone passed.");
}

function testIsSymmetric() {
	const symmetricTensor = new Tensor([
		[1, 2, 3],
		[2, 5, 6],
		[3, 6, 9],
	]);
	const nonSymmetricTensor = new Tensor([
		[1, 0, 3],
		[2, 5, 6],
		[4, 6, 9],
	]);
	assert(
		symmetricTensor.isSymmetric(),
		"Symmetric check failed for symmetric matrix"
	);
	assert(
		!nonSymmetricTensor.isSymmetric(),
		"Symmetric check failed for non-symmetric matrix"
	);
	console.log("testIsSymmetric passed.");
}

function testIsDiagonal() {
	const diagonalTensor = new Tensor([
		[1, 0, 0],
		[0, 5, 0],
		[0, 0, 9],
	]);
	const nonDiagonalTensor = new Tensor([
		[1, 2, 0],
		[0, 5, 0],
		[0, 0, 9],
	]);
	assert(
		diagonalTensor.isDiagonal(),
		"Diagonal check failed for diagonal matrix"
	);
	assert(
		!nonDiagonalTensor.isDiagonal(),
		"Diagonal check failed for non-diagonal matrix"
	);
	console.log("testIsDiagonal passed.");
}

function testIsOrthogonal() {
	const orthogonalTensor = new Tensor([
		[1, 0],
		[0, 1],
	]);
	const nonOrthogonalTensor = new Tensor([
		[1, 2],
		[2, 3],
	]);
	assert(
		orthogonalTensor.isOrthogonal(),
		"Orthogonal check failed for orthogonal matrix"
	);
	assert(
		!nonOrthogonalTensor.isOrthogonal(),
		"Orthogonal check failed for non-orthogonal matrix"
	);
	console.log("testIsOrthogonal passed.");
}

function testIsIdentity() {
	const identityTensor = Tensor.identity(3, "float64");
	const nonIdentityTensor = new Tensor([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 0],
	]);
	assert(
		identityTensor.isIdentity(),
		"Identity check failed for identity matrix"
	);
	assert(
		!nonIdentityTensor.isIdentity(),
		"Identity check failed for non-identity matrix"
	);
	console.log("testIsIdentity passed.");
}

function testTranspose() {
	const data = [
		[1, 2],
		[3, 4],
	];
	const tensor = new Tensor(data);
	const transposed = tensor.transpose();
	const expected = new Float64Array([1, 3, 2, 4]);
	assert(arraysEqual(transposed.data, expected), "Transpose failed");
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
	const expected = new Float64Array([4, 4, 10, 8]);
	assert(arraysEqual(product.data, expected), "Matrix multiplication failed");
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
	const expected = new Float64Array([4, 4, 10, 8]); // Flat array to match TypedArray output
	assert(
		arraysEqual(product.data, expected),
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
	const expectedL = new Float64Array([1, 0, 0, 4, 1, 0, 4, 0.5, 1]); // Flat array
	const expectedU = new Float64Array([1, 2, 2, 0, -4, -6, 0, 0, -1]);
	assert(
		arraysEqual(L.data, expectedL) && arraysEqual(U.data, expectedU),
		"LU decomposition failed"
	);
	console.log("testLUdecompose passed.");
}

function testQRDecomposition() {
	const data = [
		[12, -51, 4],
		[6, 167, -68],
		[-4, 24, -41],
	];
	const tensor = new Tensor(data);
	const { Q, R } = tensor.qrDecompose();

	// Log calculated Q and R matrices
	console.log("Calculated Q:", Q.data, "Shape:", Q.shape);
	console.log("Calculated R:", R.data, "Shape:", R.shape);

	// Verify Q is orthogonal: Q * Q^T = I
	const identityExpected = Tensor.identity(Q.shape[0], Q.dtype);
	const QQt = Q.matMul(Q.transpose());

	// Log the matrix Q * Q^T to check if it is the identity matrix
	console.log("Q * Q^T (Should be Identity):", QQt.data);

	assert(
		arraysAlmostEqual(QQt.data, identityExpected.data, 1e-6),
		"Q is not orthogonal"
	);

	// Verify R is upper triangular
	let upperTriangular = true;
	for (let i = 0; i < R.shape[0]; i++) {
		for (let j = 0; j < i; j++) {
			if (Math.abs(R.data[i * R.shape[1] + j]) > 1e-6) {
				console.log(
					`R[${i},${j}] is not zero as expected: `,
					R.data[i * R.shape[1] + j]
				);
				upperTriangular = false;
			}
		}
	}
	assert(upperTriangular, "R is not upper triangular");

	console.log("testQRDecomposition passed.");
}
function testQRAlgorithm() {
	// Use a simple symmetric matrix for which eigenvalues and eigenvectors are straightforward to compute
	const data = [
		[2, -1, 0],
		[-1, 2, -1],
		[0, -1, 2],
	];
	const tensor = new Tensor(data);
	const { eigenvalues, eigenvectors } = tensor.qrAlgorithm();

	// Known eigenvalues for this tridiagonal matrix are approximately [3.414, 2, 0.586]
	const expectedEigenvalues = new Float64Array([3.414, 2, 0.586]).sort(
		(a, b) => a - b
	);
	const computedEigenvalues = new Float64Array(eigenvalues).sort(
		(a, b) => a - b
	);
	console.log("expected eigenvalues:", expectedEigenvalues);
	console.log("Computed eigenvalues:", computedEigenvalues);
	console.log("Computed eigenvectors:", eigenvectors);

	// Checking the accuracy of eigenvalues
	assert(
		arraysAlmostEqual(computedEigenvalues, expectedEigenvalues, 1e-3),
		"Eigenvalues do not match expected values"
	);

	// Verify that eigenvectors are orthogonal
	const QQt = eigenvectors.matMul(eigenvectors.transpose());
	const identity = Tensor.identity(eigenvectors.shape[0], tensor.dtype);
	assert(
		arraysAlmostEqual(QQt.data, identity.data, 1e-4),
		"Eigenvectors are not orthogonal"
	);

	console.log("testQRAlgorithm passed.");
}

function testInverse() {
	const data = [
		[4, 7],
		[2, 6],
	];
	const tensor = new Tensor(data);
	const inverse = tensor.inverse();
	const expected = new Float64Array([0.6, -0.7, -0.2, 0.4]);

	// Use arraysAlmostEqual for comparison
	assert(
		arraysAlmostEqual(inverse.data, expected),
		"Inverse calculation failed"
	);

	console.log("testInverse passed.");
}

function testSVDandPseudoInverse() {
	const tensor = new Tensor([
		[1, 2, 3],
		[4, 5, 6],
	]);

	const { U, Sigma, V } = tensor.svd();
	const pseudoInverse = tensor.pseudoInverse();

	console.log("U * U^T (Should be identity):", U.matMul(U.transpose()).data);
	console.log("V * V^T (Should be identity):", V.matMul(V.transpose()).data);
	console.log("Sigma:", Sigma.data);
	console.log("Pseudo-Inverse:", pseudoInverse.data);

	// Additional check to see if U, Sigma, V can reconstruct the original matrix
	const reconstructed = U.matMul(Sigma).matMul(V.transpose());
	console.log("Reconstructed Matrix:", reconstructed.data);
	console.log("Original Matrix:", tensor.data);
}

function testSVD() {
    const matrix = new Tensor([
        [1, 2],
        [2, 3]
    ]);
    const { U, Sigma, V } = matrix.svd();

    // Check orthogonality of U and V
    const identityU = U.matMul(U.transpose());
    const identityV = V.matMul(V.transpose());
    console.log("Identity from U*U^T:", identityU.data);
    console.log("Identity from V*V^T:", identityV.data);

    // Check for identity matrices
    assert(identityU.isIdentity(), "U is not orthogonal");
    assert(identityV.isIdentity(), "V is not orthogonal");

    // Reconstruct original matrix
    const reconstructed = U.matMul(Sigma).matMul(V.transpose());
    console.log("Reconstructed Matrix:", reconstructed.data);
    console.log("Original Matrix:", matrix.data);

    // Check if reconstructed matrix matches the original
    assert(arraysAlmostEqual(reconstructed.data, matrix.data, 1e-4), "SVD reconstruction does not match the original matrix");
}

function testPseudoInverse() {
	const data = [
		[1, 2, 3],
		[4, 5, 6],
	];
	const tensor = new Tensor(data);
	const pseudoInverse = tensor.pseudoInverse();
	const expectedPseudoInverse = new Float64Array([
		// Calculated using a reliable numerical tool like MATLAB or NumPy
		-0.9444444444444444, 0.8888888888888888, -0.2222222222222222,
		0.1111111111111111, 0.7222222222222222, -0.4444444444444444,
	]);

	// Debugging output
	console.log("Calculated Pseudo-Inverse:", pseudoInverse.data);
	console.log("Expected Pseudo-Inverse:", expectedPseudoInverse);

	// Check if the calculated pseudo-inverse matches the expected values
	assert(
		arraysAlmostEqual(
			pseudoInverse.data,
			expectedPseudoInverse.map((x) => parseFloat(x.toFixed(5))),
			1e-2
		),
		"Pseudo-inverse calculation failed"
	);

	console.log("testPseudoInverse passed.");
}

function testNormalize() {
	const data = [1, 2, 3, 4, 5];
	const tensor = new Tensor(data);
	const normalized = tensor.normalize();
	const expected = new Float64Array([0, 0.25, 0.5, 0.75, 1.0]);
	assert(
		arraysEqual(
			normalized.data,
			expected.map((x) => parseFloat(x.toFixed(2)))
		),
		"Normalization failed"
	);
	console.log("testNormalize passed.");
}

function testStandardize() {
	const data = [1, 2, 3, 4, 5];
	const tensor = new Tensor(data);
	const standardized = tensor.standardize();
	assert(
		Math.abs(standardized.mean()) < 1e-10, // Checking mean close to zero due to potential floating-point precision issues
		"Standardization failed to normalize mean"
	);
	//console.log("Standardized Data:", standardized.data);
	console.log("testStandardize passed.");
}

function testReshape() {
	const data = [1, 2, 3, 4, 5, 6];
	const tensor = new Tensor(data);
	const reshaped = tensor.reshape([2, 3]);
	const expected = new Float64Array([1, 2, 3, 4, 5, 6]);
	assert(arraysEqual(reshaped.data, expected), "Reshape failed");
	console.log("testReshape passed.");
}

function runAllTests() {
	testTensorCreation();

	testAdd();
	testSubtract();
	testMultiply();
	testDivide();
	testMax();
	testMin();
	testSum();
	testKahanSum();
	testIsEmpty();
	testFill();
	testClone();
	testIsSymmetric();
	testIsDiagonal();
	testIsOrthogonal();
	testTranspose();
	testMatMul();
	testStrassenMatMul();
	testLUdecompose();
	testInverse();
	testQRDecomposition();
	testQRAlgorithm();
	testSVD();
	testSVDandPseudoInverse();
	testPseudoInverse();

	testNormalize();
	testStandardize();
	testReshape();
}

runAllTests();

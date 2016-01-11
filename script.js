$(document).ready(function () {
    var rows;
    var cols;
    var id;
    var moveLeft = 0;
    var moveDown = 0;

    function formatLatex(matrix, name)
    {
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix[0].length; j++)
        {
          matrix[i][j] = Math.round(10000*matrix[i][j])/10000;
        }
      }
      var output = name + "=";
      output+= "\\left[ \\begin{array}{"
      for(var i = 0; i < matrix[0].length; i++)
      {
        output += "c";
      }
      output += "}\n"

      for(var i = 0; i < matrix.length; i++)
      {
        output += matrix[i].join(" & ");
        if(i == matrix.length - 1)
        {
          output += "\\end{array} \\right]"
        }
        else {
          output += "\\\\ \n"
        }
      }
      return output;
    }
    function displayLUDecomposition(matrix)
    {
      var obj = LUDecomposition(matrix);
      var L = obj.L;
      var U = obj.U;
      var P = obj.P;
      var latexL = "$$" + formatLatex(L, "L") + "$$";
      var latexU = "$$" + formatLatex(U, "U") + "$$";
      var result = latexL + latexU;
      if(P !== undefined)
      {
        var permutation = "$$" + formatLatex(P, "P") + "$$";
        result = permutation + result;
      }
      $('#output').html(result);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function isPositiveDefiniteSymmetric(matrix)
    {
      var transpose = numeric.transpose(matrix);
      if(matrix.length != matrix[0].length)
      {
        return false;
      }
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix.length; j++)
        {
          if(matrix[i][j] != transpose[i][j])
          {
            return false;
          }
        }
      }
      var eigenvaluesRes = numeric.eig(matrix);
      for(var i = 0; i < eigenvaluesRes.lambda.x.length; i++)
      {
        if(eigenvaluesRes.lambda.x[i] <= 0)
        {
          return false;
        }
      }
      return true;
    }
    function findRowWithMostZeros(matrix)
    {
      var rowWithMostZeros = 0;
      var max = 0;

      for(var i = 0; i < matrix.length; i++)
      {
        var zeroCounter = 0;
        for(var j = 0; j < matrix[0].length; j++)
        {
          if(matrix[i][j] == 0)
          {
            zeroCounter++;
          }
        }
        if(zeroCounter > max)
        {
          max = zeroCounter;
          rowWithMostZeros = i;
        }
      }
      return rowWithMostZeros;
    }

    function determinant(matrix, rowToExpand)
    {
      var det = 0;
      if(matrix.length != 2)
      {
        for(var i = 0; i < matrix[rowToExpand].length; i++)
        {
          if(matrix[rowToExpand][i] == 0)
          {
            continue;
          }
          else {
            var newMatrix = JSON.parse(JSON.stringify(matrix));
            newMatrix.splice(rowToExpand, 1);
            newMatrix = numeric.transpose(newMatrix);
            newMatrix.splice(i, 1);
            newMatrix = numeric.transpose(newMatrix);
            var rowWithMostZeros = findRowWithMostZeros(newMatrix);
            det += Math.pow(-1, rowToExpand + i)*matrix[rowToExpand][i]*determinant(newMatrix, rowWithMostZeros);
          }

        }
        return det;
      }
      else {
        return matrix[1][1]*matrix[0][0] - matrix[1][0]*matrix[0][1];
      }
    }

    function smartDeterminantFinder(matrix)
    {
      var det = 1;
      var decomposition = LUDecomposition(matrix);
      for(var matrix in decomposition)
      {
        var rowWithMostZeros = findRowWithMostZeros(decomposition[matrix]);
        det *= determinant(decomposition[matrix], rowWithMostZeros);
      }
      return det;
    }

    function displayDeterminant(matrix)
    {
      if(matrix[0].length != matrix.length || matrix.length == 1)
      {
        $('#output').html("<span class='error centerTheButton'>Determinant doesn't exist for this matrix<span>");
      }
      else {
        $('#output').html("$$Determinant = " + smartDeterminantFinder(matrix) + "$$");
      }
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function CholeskyFactorization(matrix, solutions, sizeOfOriginal)
    {
      if(matrix.length == 1 && matrix[0].length == 1)
      {
        solutions.push(Math.pow(matrix[0][0],0.5));
        var solvedMatrix = []
        for(var i = 0; i < sizeOfOriginal; i++)
        {
            solvedMatrix.push([]);
        }
        for(var i = 0; i < sizeOfOriginal; i++)
        {
          for(var j = 0; j < sizeOfOriginal; j++)
          {
            solvedMatrix[i][j] = 0;
          }
        }

        for(var i = 0; i < sizeOfOriginal; i++)
        {
          for(var j = i; j < sizeOfOriginal; j++)
          {
            solvedMatrix[i][j] = solutions[0];
            solutions.splice(0,1);
          }
        }

        solvedMatrix = numeric.transpose(solvedMatrix);
        transposedSolvedMatrix = numeric.transpose(solvedMatrix);
        var latexL = "$$" + formatLatex(solvedMatrix, "L") + "$$"
        var latexL_t = "$$" + formatLatex(transposedSolvedMatrix, "L^T") + "$$"
        $('#output').html(latexL +  " " + latexL_t);
        MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
      }
      else {
        var firstEntry = Math.pow(matrix[0][0], 0.5);
        var firstColSolved = [];
        for(var i = 0; i < 1; i++)
        {
          for(var j = 0; j < matrix.length; j++)
          {
            firstColSolved.push(matrix[j][i]/firstEntry);
            solutions.push(matrix[j][i]/firstEntry);
          }
        }
        firstColSolved.splice(0,1);

        var innerMatrix = [];
        for(var i = 1; i < matrix.length; i++)
        {
          var arr = [];
          for(var j = 1; j < matrix.length; j++)
          {
            arr.push(matrix[i][j]);
          }
          innerMatrix.push(arr);
        }
        var subtractionPart = [];
        for(var i = 0; i < firstColSolved.length; i++)
        {
          subtractionPart.push([]);
        }

        for(var i = 0; i < firstColSolved.length; i++)
        {
          for(var j = 0; j < firstColSolved.length; j++)
          {
            subtractionPart[i][j] =  firstColSolved[i]*firstColSolved[j];
          }
        }
        var computedInnerMatrix = numeric.sub(innerMatrix, subtractionPart);
        CholeskyFactorization(computedInnerMatrix, solutions, sizeOfOriginal);
      }

    }

    function SVDCalculator(matrix)
    {
      var eigenValueStructure = numeric.eig(numeric.dot(numeric.transpose(matrix),matrix));
      var eigenpairs = [];
      for(var i = 0; i < eigenValueStructure.lambda.x.length; i++)
      {
        var eigenpair = {};
        eigenpair.lambda = eigenValueStructure.lambda.x[i];
        eigenpair.vector = numeric.transpose(eigenValueStructure.E.x)[i];
        eigenpairs.push(eigenpair);
      }
      eigenpairs.sort(function(a,b){return b.lambda - a.lambda})

      var V_t = [];
      for(var i = 0; i < eigenpairs.length; i++)
      {
        V_t.push(eigenpairs[i].vector);
      }
      var V = numeric.transpose(V_t);
      var S = []
      for(var i = 0; i < matrix.length; i++)
      {
        var arr = [];
        for(var j = 0; j < matrix[0].length; j++)
        {
          arr.push(0);
        }
        S.push(arr);
      }
      var limit;
      if(matrix[0].length > matrix.length)
      {
        limit = matrix.length;
      }
      else {
        limit = matrix[0].length;
      }
      var singularValues = [];
      for(var i = 0; i < limit; i++)
      {
        if(isNaN(Math.pow(eigenpairs[i].lambda,0.5)) || eigenpairs[i].lambda < Math.pow(10,-10))
        {
          S[i][i] = 0
        }
        else {
          S[i][i] = Math.pow(eigenpairs[i].lambda,0.5);
        }
        singularValues.push(S[i][i]);
      }

      var Av_t = numeric.transpose(numeric.dot(matrix,V));
      var U = [];
      for(var i = 0; i < matrix.length; i++)
      {
        if(singularValues.length == 0 || singularValues[0] == 0)
        {
          var newMat;
          while(true)
          {
            newMat = JSON.parse(JSON.stringify(U));
            var newRow = [];
            for(var j = 0; j < Av_t[0].length; j++)
            {
              newRow.push(Math.random());
            }
            newMat[newMat.length] = newRow;
            var max;
            if(newMat[0].length < newMat.length)
            {
              max = newMat[0].length;
            }
            else {
              max = newMat.length;
            }
            if(rank(JSON.parse(JSON.stringify(newMat))) == max);
            {
              break;
            }
          }
          U = newMat;
        }
        else {
          U[i] = numeric.div(Av_t[i], singularValues[0]);
        }
        singularValues.splice(0,1);
      }
      U = numeric.transpose(U);
      var columns = [];
      for(var i = 0; i < U[0].length; i++)
      {
        var arr = []
        for(var j = 0; j < U.length; j++)
        {
          arr.push(U[j][i]);
        }
        columns.push(arr);
      }

      var orthogonalizedVectors = [];
      localStorage.clear();
      for(var i = 0; i < columns.length; i++)
      {
        var projectionComponent = [];
        for(var k = 0; k < matrix.length; k++)
        {
          projectionComponent.push(0);
        }
        for(var j = 0; j < i; j++)
        {
          projectionComponent = numeric.add(projectionComponent, projection(orthogonalizedVectors[j], columns[i]));
        }
        var tempU = numeric.sub(columns[i], projectionComponent);
        localStorage['' + i] = JSON.stringify(tempU.slice(0));
        orthogonalizedVectors = [];
        for(var l = 0; l <= i; l++)
        {
          orthogonalizedVectors.push(JSON.parse(localStorage['' + l]));
        }
      }

      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        var norm = numeric.norm2(orthogonalizedVectors[i])
        for(var j = 0; j < orthogonalizedVectors[i].length; j++)
        {
          orthogonalizedVectors[i][j] /= norm;
        }
      }
      var Q = [];
      for(var k = 0; k < orthogonalizedVectors.length; k++)
      {
        Q.push([]);
      }
      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        for(var j = 0; j < orthogonalizedVectors[i].length; j++)
        {
          Q[i][j] = orthogonalizedVectors[i][j];
        }
      }
      U = numeric.transpose(Q);

      var latexU = "$$" + formatLatex(U, "U") + "$$";
      var latexSigma = "$$" + formatLatex(S, "\\sum") + "$$";
      var latexV_t= "$$" + formatLatex(V_t, "V^T") + "$$";
      $('#output').html(latexU + latexSigma + latexV_t);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function projection(u, v)
    {
      var coefficient = numeric.dot(v,u)/numeric.dot(u,u);
      for(var i = 0; i < u.length; i++)
      {
        u[i] = coefficient*u[i];
      }
      return u;
    }

    function QRDecomposition(matrix)
    {
      for(var i = 0; i < matrix[0].length; i++)
      {
        for(var j = 0; j < cols; j++)
        {
          matrix[i][j] = parseFloat(matrix[i][j]);
        }
      }
      var columns = [];
      for(var i = 0; i < matrix[0].length; i++)
      {
        var arr = []
        for(var j = 0; j < matrix.length; j++)
        {
          arr.push(matrix[j][i]);
        }
        columns.push(arr);

      }

      var orthogonalizedVectors = [];
      localStorage.clear();
      for(var i = 0; i < columns.length; i++)
      {
        var projectionComponent = [];
        for(var k = 0; k < matrix.length; k++)
        {
          projectionComponent.push(0);
        }
        for(var j = 0; j < i; j++)
        {
          projectionComponent = numeric.add(projectionComponent, projection(orthogonalizedVectors[j], columns[i]));
        }
        var tempU = numeric.sub(columns[i], projectionComponent);
        localStorage['' + i] = JSON.stringify(tempU.slice(0));
        orthogonalizedVectors = [];
        for(var l = 0; l <= i; l++)
        {
          orthogonalizedVectors.push(JSON.parse(localStorage['' + l]));
        }
      }

      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        var norm = numeric.norm2(orthogonalizedVectors[i])
        for(var j = 0; j < orthogonalizedVectors[i].length; j++)
        {
          orthogonalizedVectors[i][j] /= norm;
        }
      }
      var Q = [];
      for(var k = 0; k < orthogonalizedVectors.length; k++)
      {
        Q.push([]);
      }
      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        for(var j = 0; j < orthogonalizedVectors[i].length; j++)
        {
          Q[i][j] = orthogonalizedVectors[i][j];
        }
      }
      Q = numeric.transpose(Q);
      var R = [];
      for(var k = 0; k < orthogonalizedVectors.length; k++)
      {
        R.push([]);
      }

      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        for(var j = 0; j < orthogonalizedVectors.length; j++)
        {
          R[i][j] = 0;
        }
      }

      for(var i = 0; i < orthogonalizedVectors.length; i++)
      {
        for(var j = i; j < orthogonalizedVectors.length; j++)
        {
          R[i][j] = numeric.dot(columns[j], orthogonalizedVectors[i]);
        }
      }
      var latexQ = "$$" + formatLatex(Q,"Q") + "$$";
      var latexR = "$$" + formatLatex(R,"R") + "$$";
      $('#output').html(latexQ + " " + latexR);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function getEigenValues(matrix)
    {
      var eigenvalueResults = numeric.eig(matrix);
      var allEigenpairs = "";
      var allEigenvectors = numeric.transpose(numeric.eig(matrix).E.x);
      for(var i = 0; i < eigenvalueResults.lambda.x.length; i++)
      {
          allEigenpairs += "$$\\lambda_" + (i+1) + "= " + Math.round(10000*eigenvalueResults.lambda.x[i])/10000 + ", ";
          var eigenvector = numeric.transpose([allEigenvectors[i]]);
          var latexEigenVector =  formatLatex(eigenvector, "v_" + (i+1));
          allEigenpairs += latexEigenVector + "$$ ";
      }
      $('#output').html(allEigenpairs);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function rank(matrix)
    {
      var pivot = 0;
      for(var i = 0; i < matrix.length; i++)
      {
        var hasSwitched = false;
        while(pivot <= matrix.length - 1 && matrix[i][pivot] == 0)
        {
          for(var j = i+1; j < matrix.length; j++)
          {
            if(matrix[j][pivot] != 0)
            {
              var tempArr = matrix[i];
              matrix[i] = matrix[j];
              matrix[j] = tempArr;
              hasSwitched = true;
              break;
            }
          }
          if(!hasSwitched)
          {
            pivot++;
          }
        }

        if(pivot <= matrix[0].length - 1)
        {
          if(matrix[i][pivot] != 0)
          {
            var divider = matrix[i][pivot];
            for(var k = 0; k < matrix[i].length; k++)
  					{
  						matrix[i][k] /= divider;
  					}
          }

        for(var j = 0; j < matrix.length; j++)
  		  {
          if(j == i)
  				{
            continue;
  				}
  			  else
  				{
  					if(matrix[j][pivot] != 0)
  					{
  						var newRow = [];
  						for(var k = 0; k < matrix[j].length; k++)
  						{
  								newRow[k] = matrix[j][k] - matrix[i][k]*matrix[j][pivot];
  						}
  						matrix[j] = newRow;
  					}
  				}
  			}
        pivot++;
      }
      }
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix[0].length; j++)
        {
          if(matrix[i][j] == -0)
          matrix[i][j] = 0;
        }
      }
      var i = 0;
      var rank = 0;
      for(var j = 0; j < matrix[0].length; j++)
      {
        if(i < matrix.length && matrix[i][j] == 1)
        {
          i++;
          rank++;
        }
      }
      return rank;
    }

    function RREF(matrix)
    {
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix[0].length; j++)
        {
          matrix[i][j] = parseInt(matrix[i][j]);
        }
      }
      var pivot = 0;
      for(var i = 0; i < matrix.length; i++)
      {
        var hasSwitched = false;
        while(pivot <= matrix.length - 1 && matrix[i][pivot] == 0)
        {
          for(var j = i+1; j < matrix.length; j++)
          {
            if(matrix[j][pivot] != 0)
            {
              var tempArr = matrix[i];
              matrix[i] = matrix[j];
              matrix[j] = tempArr;
              hasSwitched = true;
              break;
            }
          }
          if(!hasSwitched)
          {
            pivot++;
          }
        }

        if(pivot <= matrix[0].length - 1)
        {
          if(matrix[i][pivot] != 0)
          {
            var divider = matrix[i][pivot];
            for(var k = 0; k < matrix[i].length; k++)
  					{
  						matrix[i][k] /= divider;
  					}
          }

        for(var j = 0; j < matrix.length; j++)
  		  {
          if(j == i)
  				{
            continue;
  				}
  			  else
  				{
  					if(matrix[j][pivot] != 0)
  					{
  						var newRow = [];
  						for(var k = 0; k < matrix[j].length; k++)
  						{
  								newRow[k] = matrix[j][k] - matrix[i][k]*matrix[j][pivot];
  						}
  						matrix[j] = newRow;
  					}
  				}
  			}
        pivot++;
      }
      }
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix[0].length; j++)
        {
          if(matrix[i][j] == -0)
          matrix[i][j] = 0;
        }
      }
      var latexRREF = "$$" + formatLatex(matrix,"RREF") + "$$";
      var i = 0;
      var rank = 0;
      for(var j = 0; j < matrix[0].length; j++)
      {
        if(i < matrix.length && matrix[i][j] == 1)
        {
          i++;
          rank++;
        }
      }
      var nullity = matrix[0].length - rank;

      latexRREF += "$$Rank = " + rank + ", Nullity = " + nullity + "$$";
      $('#output').html(latexRREF);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function LUDecomposition(matrix)
    {
      var operations = [];
      var permutation;
      var limit;
      var L = [];
      for(var i = 0; i < matrix.length; i++)
      {
        var row = [];
        for(var j = 0; j < matrix.length; j++)
        {
          if(i==j)
          {
            row.push(1);
          }
          else {
            row.push(0);
          }
        }
        L.push(row);
      }
      permutation = JSON.parse(JSON.stringify(L));
      var PL= JSON.parse(JSON.stringify(L));
      var identity= JSON.parse(JSON.stringify(L));

      if(matrix[0].length < matrix.length)
      {
        limit = matrix[0].length;
      }
      else {
        limit = matrix.length;
      }
      for(var i = 0; i < limit; i++)
      {
        for(var j = i+1; j < matrix.length; j++)
        {
          if(matrix[i][i] != 0)
          {
            var multiplier = matrix[j][i]/matrix[i][i];
            matrix[j] = numeric.sub(matrix[j], numeric.mul(matrix[i], multiplier));
            operations.push({"isSwap": false, "rowModified": j, "rowSubtractedFrom": i, "multiplier": multiplier});
          }
          else
          {
            var temp = matrix[i];
            matrix[i] = matrix[j];
            matrix[j] = temp;
            var temp2 = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = temp2;
            operations.push({"isSwap": true, "swap1" : i, "swap2" : j});
          }
        }
      }


      for(var i = operations.length - 1; i >= 0; i--)
      {
        if(operations[i].isSwap)
        {
            var arr = PL[operations[i].swap1]
            PL[operations[i].swap1] = PL[operations[i].swap2];
            PL[operations[i].swap2] = arr;
        }
        else
        {
          PL[operations[i].rowModified] = numeric.add(PL[operations[i].rowModified], numeric.mul(PL[operations[i].rowSubtractedFrom], operations[i].multiplier))
        }
      }

      L = numeric.dot(numeric.inv(permutation), PL);
      var obj = {};
      obj.L = L;
      obj.U = matrix;
      var isPIdentity = true;
      for(var i = 0; i < permutation.length; i++)
      {
        if(numeric.eq(permutation[i], identity[i]).indexOf(false) != -1)
        {
          isPIdentity = false;
        }
      }
      if(!isPIdentity)
      {
        obj.P = permutation;
      }
      return obj;
    }

    function inverse(matrix)
    {
      var inverse = numeric.inv(matrix);
      var inverseLatex = "$$" + formatLatex(numeric.inv(matrix), "A^{-1}") + "$$";
      $('#output').html(inverseLatex);
      MathJax.Hub.Queue(["Typeset",MathJax.Hub,"MyEquation"]);
    }

    function isValidMatrix(matrix)
    {
      for(var i = 0; i < matrix.length; i++)
      {
        for(var j = 0; j < matrix[0].length; j++)
        {
          if(isNaN(matrix[i][j]))
          {
            return false;
          }
        }
      }
      var columns = matrix[0].length;
      for(var i = 0; i < matrix.length; i++)
      {
        if(columns != matrix[i].length)
        {
          return false;
        }
      }
      return true;

    }

    $('.dropdown-content').find('a').click(function(event)
    {
      var source = $('#DimensionSubmission').html();
      var matrixDimensionsTemplate = Handlebars.compile(source);
      var noTextArea = false;
      if($('#matrixData').length == 0)
      {
        noTextArea = true;
      }
      var data = {id: $(this).html(), hasNoTextArea: noTextArea};
      if(noTextArea)
      {
        $('#optionType').remove();
        $('#rightPane').append(matrixDimensionsTemplate(data));
      }
      else
      {
        $('#optionType').html($(this).html());
      }
      id = $(this).html();

      $('#submitDimensions').click(function(event)
      {
        event.preventDefault();
        var rows = document.getElementById('matrixData').value.split('\n');
        var matrix = [];
        while(rows.indexOf("") != -1)
        {
          rows.splice(rows.indexOf(""), 1);
        }
        rows.forEach(function(element)
        {
          var arr = element.split(" ");
          while(arr.indexOf("") != -1)
          {
            arr.splice(arr.indexOf(""), 1);
          }
          matrix.push(arr);
        });
        var specificSquareRequirement = false;
        switch(id)
        {
          case "Cholesky Decomposition":
          case  "Eigenvalues":
          case "Determinant":
          case "Inverse":
            specificSquareRequirement = true;
            break;
          default:
            break;

        }
        if(!isValidMatrix(matrix))
        {
          var numberErrorMatrixTemplate = $('#NumberErrorMatrix').html();
          var renderNumberErrorMatrix = Handlebars.compile(numberErrorMatrixTemplate);
          $('#modifiedMatrixPart').html(renderNumberErrorMatrix());
        }
        else if (specificSquareRequirement && (matrix.length != matrix[0].length)) {
          var incorrectMatrixTemplate = $('#IncorrectMatrix').html();
          var renderIncorrectMatrix = Handlebars.compile(incorrectMatrixTemplate);
          $('#modifiedMatrixPart').html(renderIncorrectMatrix({id: id}));
        }
        else if(matrix.length > 10 || matrix[0].length > 10)
        {
          var tooBigDimensionsTemplate = $('#GreaterThanRequired').html();
          var renderTooBigDimensions = Handlebars.compile(tooBigDimensionsTemplate);
          $('#modifiedMatrixPart').html(renderTooBigDimensions());
        }
        else {
          var outputTemplate = $('#Output').html();
          var renderOutputTempate = Handlebars.compile(outputTemplate);
          $('#modifiedMatrixPart').html(renderOutputTempate());
              var parsedMatrix = [];
              for(var i = 0; i < matrix.length; i++)
              {
                var arr = [];
                for(var j = 0; j < matrix[0].length; j++)
                {
                  arr.push(parseFloat(matrix[i][j]));
                }
                parsedMatrix.push(arr);
              }
              switch(id)
              {
                case "LU Decomposition":
                  displayLUDecomposition(parsedMatrix);
                  break;
                case "QR Decomposition":
                  QRDecomposition(parsedMatrix);
                  break;
                case "RREF":
                  RREF(parsedMatrix);
                  break;
                case "Inverse":
                var copy = JSON.parse(JSON.stringify(parsedMatrix));
                if(smartDeterminantFinder(parsedMatrix) == 0)
                {
                  var notInvertibleTemplate = $('#NotInvertible').html();
                  var renderNotInvertibleTemplate = Handlebars.compile(notInvertibleTemplate);
                  $('#output').html(renderNotInvertibleTemplate());
                }
                else {
                  inverse(copy);
                }
                break;
                case "Determinant":
                  displayDeterminant(parsedMatrix);
                  break;
                case "Eigenvalues":
                    getEigenValues(parsedMatrix);
                  break;
                case "Cholesky Decomposition":
                  if(isPositiveDefiniteSymmetric(parsedMatrix))
                  {
                    CholeskyFactorization(parsedMatrix,[], parsedMatrix.length);
                  }
                  else {
                    var incompatibleWithCholeskyTemplate = $('#incompatibleWithCholesky').html();
                    var renderIncompatibleWithCholeskyTemplate = Handlebars.compile(incompatibleWithCholeskyTemplate);
                    $('#output').html(renderIncompatibleWithCholeskyTemplate());
                  }
                  break;
                case "Singular Value Decomposition":
                    SVDCalculator(parsedMatrix);
                    break;
              }
          }
      });
    });
});

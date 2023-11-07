pragma circom 2.0.0;

template InnerProd () {  

   // Declaration of signals 
   signal input input_A[3];  
   signal input input_B[3];  
   signal output ip;

   var sum = 0;

   for (var i = 0; i < 3; i++) {
      sum = sum + input_A[i]*input_B[i];
   }

   ip <== sum;
}

component main = InnerProd();

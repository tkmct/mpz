pragma circom 2.0.0;

/*This circuit template checks that c is the multiplication of a and b.*/  

template InnerProd () {  

   // Declaration of signals 
   signal input input_A[10];  
   signal input input_B[10];  
   signal output ip;

   // var sum = 0;
   // for (var i = 0; i < 10; i++) {
      // temp = temp + input_A[i]*input_B[i];
   // }
   // var temp = input_A[0] * input_B[0];
   // sum = sum + temp;
   // var sum1 = input_A[0] * input_B[0];
   // var sum2 = sum + sum1;
   // sum = sum2;

   var sum = 0;
   sum = sum + input_A[3] * input_B[8];

   ip <== sum;

   // signal input garb_a;  
   // signal input eval_b;  
   // var i;
   // signal output c;  
   

   // Computation  
   // i <== garb_a + eval_b;

   // var j = 100;

   // var k = 0;

   // if (garb_a != 3) {
   //    k = k + 1;
   // } else {
   //    j = j + 1;
   // }

   // var t = i * j;
   // c <== garb_a * t;
   // i = garb_z[1];
   // i = garb_a + eval_b;
   // i = garb_a * eval_b;
   // i = garb_a + 101;
   // i = eval_b * 20;
   // c <== i * i;
}

component main = InnerProd();

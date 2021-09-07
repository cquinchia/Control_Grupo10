
/**
 *
 * @author Cristhian Becerra
 */
public class Aseguradora {
    
    public static double liquidarPrestaciones(int salario, int diasTrabajados){
	double sal;
        
        if(salario<=908526*2){       
            salario = salario+106454;
            sal=salario-106454;
        }
        else{
            sal=salario;
        }
        double primaServicios=salario*diasTrabajados/360;
	double cesantias=salario*diasTrabajados/360;
	double interesesCesantias=0.12*cesantias;		
	double vacaciones=sal*diasTrabajados/720;            
            
	return Math.ceil(primaServicios+cesantias+interesesCesantias+vacaciones);	
    }  

    public static double liquidarSegSocial(int salario, int diasTrabajados){

        double salud=salario*0.085*12*diasTrabajados/360;
	double pension=salario*0.12*12*diasTrabajados/360;
	double riesgo=salario*0.00522*12*diasTrabajados/360;		
            
	return Math.ceil(salud+pension+riesgo);	
    }
}
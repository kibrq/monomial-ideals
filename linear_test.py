from mgraph import * 
import datetime
import subprocess
from sympy import GF, IndexedBase, prod, Symbol
import json

def generate_temp_filenames(*suffixes):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S_%f")
    return [f"tempfile_{formatted_time}_{suffix}" for suffix in suffixes]

def run_m2_json(script):
    with open(script_file,'w') as sf:
        sf.write(script)

    out_file, script_file = generate_temp_filenames('M2_output.txt','M2_script.m2')

    script = f"""out_file = "{out_file}";\n""" + script
    
    with open(script_file,'w') as sf:
        sf.write(script)

    # Prepare the command with the script and the argument
    command = [m2_command, script_file]
    # Execute the command
    result = subprocess.run(command, text=True, capture_output=True)
    # Check if the command was successful
    try:
        if result.returncode == 0:
            res = json.loads(out_file)
            os.remove(out_file)
            os.remove(script_file)

            return res
        else:
            raise Exception(result.stderr)
    except:
        raise Exception(f'M2 execution failed {script_file}')
        



def run_linear_test(graphs, vars, m2_command='M2'):
    
    in_file, out_file, script_file = generate_temp_filenames('M2_input.txt','M2_output.txt','M2_script.m2')
    write_graphs_to_file(graphs, in_file)

    script = f"""
S = ZZ/101[{','.join(vars)}];
inputFile = "{in_file}";
outputFile = "{out_file}";
"""

    script += """
isLinear = I -> (
    d := (degree I_*_0)_0;
    {d+1} == max degrees source syz gens I
)
inputLines = lines get inputFile;
filteredLines = {};
for line in inputLines do (
    I = ideal line;
    if isLinear(I) then (
        outputFile << line << endl;
    );
    
);
outputFile << close;
quit()
"""

    with open(script_file,'w') as sf:
        sf.write(script)

    
    # Prepare the command with the script and the argument
    command = [m2_command, script_file]
    # Execute the command
    result = subprocess.run(command, text=True, capture_output=True)
    # Check if the command was successful
    try:
        if result.returncode == 0:
            res = read_graphs_from_file(out_file)

            os.remove(in_file)
            os.remove(out_file)
            os.remove(script_file)

            return res
        else:
            raise Exception(result.stderr)
    except:
        raise Exception(f'M2 execution failed {script_file}')
        
        
def m2_linear_test(graphs, vars, m2_command='M2'):
    return run_linear_test(graphs, vars, m2_command=m2_command)
    
      
def syzmod(gens):
    variables = dict((v,i) for (i,v) in enumerate(list(sorted(set().union(*gens)))))
    x = IndexedBase('x')
    nvars = len(variables)
    syms = [x[i+1] for i in range(nvars)]
    kk = GF(101)
    ring = kk.old_poly_ring(*syms)
    monomials = [prod([x[variables[char]+1] for char in mon]) for mon in gens]
    # here is a hack to check linearity -- if * appears then some variable is raised to a power in the syzygy matrix.
    # otherwise, the presentation is linear
    modgens = [[x] for x in monomials]
    return ring.free_module(1).submodule(*modgens).syzygy_module()

def python_linear_test(gens):
    res = syzmod(gens)
    linear_syzgens = [gen for gen in res.gens if '*' not in str(gen)]
    linear_submod = res.submodule(*linear_syzgens)
    if linear_submod == res:
        return True
    else:
        False

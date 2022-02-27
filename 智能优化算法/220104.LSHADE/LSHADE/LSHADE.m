%%%%%%%%%%%%%%%%%%%
%% This package is a MATLAB/Octave source code of LSHADE_cnEpSin which is a new version of LSHADE-EpSin.
%% Please see the following papers:
%% 1. LSHADE_cnEpSin:
%%     Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Ensemble Sinusoidal Differential Covariance Matrix Adaptation with Euclidean Neighborhood  for Solving CEC2017 Benchmark Problems, in Proc. IEEE Congr. Evol. Comput. CEC 2017, June, Donostia - San Sebasti醤, Spain

%% 2. LSHADE-EpSin:
%%    Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan and Robert G. Reynolds: An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems, in Proc. IEEE Congr. Evol. Comput. CEC 2016, Canada, July, 2016

%% About L-SHADE, please see following papers:
%% Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.
%%  J. Zhang, A.C. Sanderson: JADE: Adaptive differential evolution with optional external archive,?IEEE Trans Evol Comput, vol. 13, no. 5, pp. 945?58, 2009

clc;
clear all;

format long;
format compact;

problem_size = 10;
max_nfes = 10000 * problem_size;

rand('seed', sum(100 * clock));

val_2_reach = 10^(-8);%%小于10^(-8)就归为0
max_region = 100.0;
min_region = -100.0;
lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];
fhd=@cec17_func;
pb = 0.4;
ps = 0.5;

% S.Ndim = problem_size;
% S.Lband = ones(1, S.Ndim)*(-100);
% S.Uband = ones(1, S.Ndim)*(100);

%%%% Count the number of maximum generations before as NP is dynamically 统计NP是动态的最大代数
%%%% decreased 下降
% G_Max = 0;
% if problem_size == 10
%     G_Max = 2163;
% end
% if problem_size == 30
%     G_Max = 2745;
% end
% if problem_size == 50
%     G_Max = 3022;
% end
% if problem_size == 100
%     G_Max = 3401;
% end

num_prbs = 30;
runs = 51;
% run_funcvals = [];
% RecordFEsFactor = ...
%     [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, ...
%     0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
% progress = numel(RecordFEsFactor);
% 
% allerrorvals = zeros(progress, runs, num_prbs);
result=zeros(num_prbs,5);

fprintf('Running LSHADE_cnEpSin on D= %d\n', problem_size)
for func = 1 : num_prbs
   % func=4;
    optimum = func * 100.0;
    S.FuncNo = func;
    
    %% Record the best results   记录最好的结果
    outcome = [];
    
    fprintf('\n-------------------------------------------------------\n')
    fprintf('Function = %d, Dimension size = %d\n', func, problem_size)
    
    for run_id = 1 : runs
        
        run_funcvals = [];
        col=1;              %% to print in the first column in all_results.mat    在all_results.mat的第一列中打印
        
        %%  parameter settings for L-SHADE                   L-SHADE的参数设置
        p_best_rate = 0.11;    %0.11
        arc_rate = 1.4;
        memory_size = 5;
        pop_size = 18 * problem_size;   %18*D
       %SEL = round(ps*pop_size);
        
        max_pop_size = pop_size;
        min_pop_size = 4.0;
        
        nfes = 0;
        %% Initialize the main population   初始化主要种群
        popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
        pop = popold; % the old population becomes the current population    老种群成为现在的种群
        
        fitness = feval(fhd,pop',func);
        fitness = fitness';%算初始种群的适应度值
        
        bsf_fit_var = 1e+30;
        bsf_index = 0;
        bsf_solution = zeros(1, problem_size);
        
        %%%%%%%%%%%%%%%%%%%%%%%% for out
        for i = 1 : pop_size
            nfes = nfes + 1;
            
            if fitness(i) < bsf_fit_var
                bsf_fit_var = fitness(i);
                bsf_solution = pop(i, :);
                bsf_index = i;
            end
            
            if nfes > max_nfes;
                break; 
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%% for out
        
        memory_sf = 0.5 .* ones(memory_size, 1);
        memory_cr = 0.5 .* ones(memory_size, 1);
        memory_pos = 1;
        
        archive.NP = arc_rate * pop_size; % the maximum size of the archive 档案的最大大小
        archive.pop = zeros(0, problem_size); % the solutions stored in te archive  解决方案存储在档案中
        archive.funvalues = zeros(0, 1); % the function value of the archived solutions 归档解决方案的功能价值
        
        %% main loop  主循环
        gg=0;  %%% generation counter used For Sin
        igen =1;  %%% generation counter used For LS  用于LS的一代计数器   
        goodCR = [];
        goodF = [];                
        while nfes < max_nfes
            gg=gg+1;
            
            pop = popold; % the old population becomes the current population  老种群成为现在的种群
            [temp_fit, sorted_index] = sort(fitness, 'ascend');
            
            mem_rand_index = ceil(memory_size * rand(pop_size, 1));
            mu_sf = memory_sf(mem_rand_index);
            mu_cr = memory_cr(mem_rand_index);
           
            
            %% for generating crossover rate   产生交叉率
            cr = normrnd(mu_cr, 0.1);%正态分布
            term_pos = find(mu_cr == -1);
            cr(term_pos) = 0;
            cr = min(cr, 1);
            cr = max(cr, 0);
            
            %% for generating scaling factor  用于生成比例因子
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);
            
            while ~ isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end
            sf = min(sf, 1);
            r0 = [1 : pop_size];
            popAll = [pop; archive.pop];
            [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);%随机选两个个体
            
            pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions 选择至少两个最好的解决方案
            randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]    从[1，2，3，...，pNP]中选择
            randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0 避免rand = 0，ceil（rand）= 0的问题
            pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions 随机选择顶级100p％解决方案之一
            
            vi = pop + sf(:, ones(1, problem_size)) .* (pbest - pop + pop(r1, :) - popAll(r2, :));   %公式3
            vi = boundConstraint(vi, pop, lu);%检查边界
            
                   
            mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent 掩码用于指示哪些元素来自父
            rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent 选择一个ui的元素不是来自父母的位置
            jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
            ui = vi; 
            ui(mask) = pop(mask);
                  
            children_fitness = feval(fhd, ui', func);
            children_fitness = children_fitness';
            
             bsf_fit_var_old = bsf_fit_var;
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            for i = 1 : pop_size
                nfes = nfes + 1;
                
                if children_fitness(i) < bsf_fit_var
                    bsf_fit_var = children_fitness(i);
                    bsf_solution = ui(i, :);
                    bsf_index = i;
                end
                
                if nfes > max_nfes; break; end
            end
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            
            dif = abs(fitness - children_fitness);
            
            
            %% I == 1: the parent is better; I == 2: the offspring is better
            I = (fitness > children_fitness);
            goodCR = cr(I == 1);
            goodF = sf(I == 1);
            dif_val = dif(I == 1);
            
           
            
           
            
            %      isempty(popold(I == 1, :))
            archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));
            
            [fitness, I] = min([fitness, children_fitness], [], 2);%选择
            
            run_funcvals = [run_funcvals; fitness];
            
            popold = pop;
            popold(I == 2, :) = ui(I == 2, :);
            
            num_success_params = numel(goodCR);
            
            if num_success_params > 0
                sum_dif = sum(dif_val);
                dif_val = dif_val / sum_dif;
                
                %% for updating the memory of scaling factor  用于更新比例因子的存储器
                memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
                
                %% for updating the memory of crossover rate  用于更新交叉速率的内存
                if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                    memory_cr(memory_pos)  = -1;
                else
                    memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                end
                
                memory_pos = memory_pos + 1;
                if memory_pos > memory_size;  memory_pos = 1; end
            end
            
            %% for resizing the population size 调整种群规模
            plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);
            
            if pop_size > plan_pop_size
                reduction_ind_num = pop_size - plan_pop_size;
                if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end
                
                pop_size = pop_size - reduction_ind_num;
                SEL = round(ps*pop_size);
                for r = 1 : reduction_ind_num
                    [valBest indBest] = sort(fitness, 'ascend');
                    worst_ind = indBest(end);
                    popold(worst_ind,:) = [];
                    pop(worst_ind,:) = [];
                    fitness(worst_ind,:) = [];
                end
                
                archive.NP = round(arc_rate * pop_size);
                
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1 : archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
            
        end %%%%%%%%nfes
        
        bsf_error_val = bsf_fit_var - optimum;
        if bsf_error_val < val_2_reach
            bsf_error_val = 0;
        end
        
        fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , bsf_error_val)
        outcome = [outcome bsf_error_val];
        
      
        
    end %% end 1 run
    
    fprintf('\n')
    fprintf('min error value = %1.8e, max = %1.8e, median = %1.8e, mean = %1.8e, std = %1.8e\n', min(outcome), max(outcome), median(outcome), mean(outcome), std(outcome))
    
    result(func,1)=  min(outcome);
    result(func,2)=  max(outcome);
    result(func,3)=  median(outcome);
    result(func,4)=  mean(outcome);
    result(func,5)=  std(outcome);
   
end %% end 1 function run

disp(result);

name1 = 'results_stat_';
name2 = num2str(problem_size);
name3 = '.txt';
f_name=strcat(name1,name2,name3);

save(f_name, 'result', '-ascii');

% %%% To print files
% for i =1 : num_prbs
%     name1 = 'LSHADE_cnEpSin';
%     name2 = num2str(i);
%     name3 = '_';
%     name4 = num2str(problem_size);
%     name5 = '.txt';
%     f_name=strcat(name1,name2,name3,name4,name5);
%     res = allerrorvals(:,:,i);
%     save(f_name, 'res', '-ascii');
% end

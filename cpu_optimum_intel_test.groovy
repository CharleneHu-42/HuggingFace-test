NODE_LABEL = ' '
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}
echo "NODE_LABEL: $NODE_LABEL"

model_list = ''
if ('model_list' in params) {
    echo "model_list in params"
    if (params.model_list != '') {
        model_list = params.model_list
        model_list = model_list.split(',')
    }
}
echo "model_list: $model_list"

env.token_config = ''
if ('token_config' in params) {
    echo "token_config in params"
    if (params.token_config != '') {
        env.token_config = params.token_config
    }
}
echo "token_config: $token_config"
token_config_list = token_config.split(',')
echo "token_config_list: $token_config_list"

env.batch_size = ''
if ('batch_size' in params) {
    echo "batch_size in params"
    if (params.batch_size != '') {
        env.batch_size = params.batch_size
    }
}
echo "batch_size: $batch_size"
batch_size_list = batch_size.split(',')
echo "batch_size_list: $batch_size_list"

env.decode_strategy = ''
if ('decode_strategy' in params) {
    echo "decode_strategy in params"
    if (params.decode_strategy != '') {
        env.decode_strategy = params.decode_strategy
    }
}
echo "decode_strategy: $decode_strategy"
decode_strategy_list = decode_strategy.split(',')
echo "decode_strategy_list: $decode_strategy_list"

env.test_mode = 'oi'
if ('test_mode' in params) {
    echo "test_mode in params"
    if (params.test_mode != '') {
        env.test_mode = params.test_mode
    }
}
echo "test_mode: $test_mode"
test_mode_list = test_mode.split(',')
echo "test_mode_list: $test_mode_list"

env.rank = ''
if ('rank' in params) {
    echo "rank in params"
    if (params.rank != '') {
        env.rank = params.rank
    }
}
echo "rank: $rank"
rank_list = rank.split(',')
echo "rank_list: $rank_list"

env.oi_repo = 'https://github.com/huggingface/optimum-intel'
if ('oi_repo' in params) {
    echo "oi_repo in params"
    if (params.oi_repo != '') {
        env.oi_repo = params.oi_repo
    }
}
echo "oi_repo: $oi_repo"

env.oi_branch = 'main'
if ('oi_branch' in params) {
    echo "oi_branch in params"
    if (params.oi_branch != '') {
        env.oi_branch = params.oi_branch
    }
}
echo "oi_branch: $oi_branch"

env.oi_commit = ''
if ('oi_commit' in params) {
    echo "oi_commit in params"
    if (params.oi_commit != '') {
        env.oi_commit = params.oi_commit
    }
}
echo "oi_commit: $oi_commit"

env.ipex_whl_url = ''
if ('ipex_whl_url' in params) {
    echo "ipex_whl_url in params"
    if (params.ipex_whl_url != '') {
        env.ipex_whl_url = params.ipex_whl_url
    }
}
echo "ipex_whl_url: $ipex_whl_url"

env.transformers_version = ''
if ('transformers_version' in params) {
    echo "transformers_version in params"
    if (params.transformers_version != '') {
        env.transformers_version = params.transformers_commit
    }
}
echo "transformers_commit: $transformers_version"

env.transformers_repo = 'https://github.com/huggingface/transformers'
if ('transformers_repo' in params) {
    echo "transformers_repo in params"
    if (params.transformers_repo != '') {
        env.transformers_repo = params.transformers_repo
    }
}
echo "transformers_repo: $transformers_repo"

env.transformers_branch = 'main'
if ('transformers_branch' in params) {
    echo "transformers_branch in params"
    if (params.transformers_branch != '') {
        env.transformers_branch = params.transformers_branch
    }
}
echo "transformers_branch: $transformers_branch"

env.transformers_commit = ''
if ('transformers_commit' in params) {
    echo "transformers_commit in params"
    if (params.transformers_commit != '') {
        env.transformers_commit = params.transformers_commit
    }
}
echo "transformers_commit: $transformers_commit"

env.node_http_proxy = 'http://proxy-dmz.intel.com:912'
if ('node_http_proxy' in params) {
    echo "node_http_proxy in params"
    if (params.node_http_proxy != '') {
        env.node_http_proxy = params.node_http_proxy
    }
}
echo "node_http_proxy: $node_http_proxy"

env.node_https_proxy = 'http://proxy.ims.intel.com:911'
if ('node_https_proxy' in params) {
    echo "node_https_proxy in params"
    if (params.node_https_proxy != '') {
        env.node_https_proxy = params.node_https_proxy
    }
}
echo "node_https_proxy: $node_https_proxy"

env.hf_cache = '/home/sdp/.cache/huggingface'
if ('hf_cache' in params) {
    echo "hf_cache in params"
    if (params.hf_cache != '') {
        env.hf_cache = params.hf_cache
    }
}
echo "hf_cache: $hf_cache"

env.hf_token = ' '
if ('hf_token' in params) {
    echo "hf_token in params"
    if (params.hf_token != '') {
        env.hf_token = params.hf_token
    }
}
echo "hf_token: $hf_token"

node(NODE_LABEL){
    properties(
        [disableConcurrentBuilds(),]
    )
    deleteDir()
    checkout scm
    currentBuild.displayName = "#${BUILD_NUMBER}-${NODE_LABEL}-${test_mode}"

    try{
        env.DOCKER_IMAGE = "appliedml/huggingface:cpu-base"
        stage("Prepare Env") {
            withEnv(["NODE_LABEL=${NODE_LABEL}", "oi_repo=${oi_repo}", "oi_branch=${oi_branch}", "oi_commit=${oi_commit}", \
                    "ipex_whl_url=${ipex_whl_url}", \
                    "transformers_version=${transformers_version}", "transformers_repo=${transformers_repo}", "transformers_branch=${transformers_branch}", "transformers_commit=${transformers_commit}", \
                    "node_http_proxy=${node_http_proxy}", "node_https_proxy=${node_https_proxy}", "hf_cache=${hf_cache}"]) {
                sh '''
                    set -x

                    echo  "job number: #${BUILD_NUMBER}"
                    echo  "job link: ${BUILD_URL}"

                    export http_proxy=${node_http_proxy}
                    export https_proxy=${node_https_proxy}

                    cd ${WORKSPACE}/HuggingFace/docker
                    bash build_image.sh -d cpu -t base 2>&1 | tee build_image.log
                '''

                // Start the Docker container
                testContainer = docker.image(env.DOCKER_IMAGE).run('-d', "-e http_proxy=${node_http_proxy} -e https_proxy=${node_https_proxy} -v ${WORKSPACE}:/workspace, -v ${hf_cache}:/root/.cache/huggingface/hub", "--network host --privileged")
                // Install libraries inside the container
                testContainer.inside {
                    sh '''
                        set -x

                        cd /workspace/
                        git clone ${oi_repo}
                        cd optimum-intel
                        git checkout ${oi_branch}
                        if [[ -n "${oi_commit}" ]]; then
                            git checkout ${oi_commit}
                        fi
                        pip install .

                        if [[ -n "${ipex_whl_url}" ]]; then
                            pip install ${ipex_whl_url}
                        fi

                        if [[ -n "${transformers_version}" ]]; then
                            pip install transformers==${transformers_version}
                        elif [[ -n "${transformers_commit}" ]]; then
                            cd /workspace/
                            git clone ${transformers_repo}
                            cd transformers
                            git checkout ${transformers_branch}
                            git checkout ${transformers_commit}
                            pip install .
                        fi

                        cd /workspace/HuggingFace/tests/workloads
                        pip install -r requirements.txt
                        
                    '''
                }
                // Save the container ID for later use
                env.CONTAINER_ID = testContainer.id

                archiveArtifacts artifacts: 'HuggingFace/docker/build_image.log', allowEmptyArchive: true
            }
        }

        

        stage('Run Tests') {
            withEnv(["specified_model_list=${model_list}", "token_config_list=${token_config_list}", "batch_size_list=${batch_size_list}","decode_strategy_list=${decode_strategy_list}",  "rank_list=${rank_list}", "test_mode_list=${test_mode_list}", "hf_token=${hf_token}"]) {
                // Attach to the same Docker container
                docker.image(env.DOCKER_IMAGE).inside("--volumes-from ${env.CONTAINER_ID}") {
                    sh '''
                        set -x

                        cd /workspace/HuggingFace/tests/workloads
                        echo "Test modes: ${test_mode_list}"
                        IFS=' ' read -r -a test_mode_array <<< "$test_mode_list"

                        # Check for each test mode and execute corresponding commands
                        for mode in "${test_mode_array[@]}"; do
                            if [[ "$mode" == "oi" ]]; then
                                export OI_PAGED_ATTN_BLOCK_SIZE=64
                                bash run_oi_cpu_auto.sh --optimum_intel True
                            elif [[ "$mode" == "eager" ]]; then
                                bash run_oi_cpu_auto.sh
                            elif [[ "$mode" == "compile" ]]; then
                                bash run_oi_cpu_auto.sh -c True
                            fi
                        done
                    '''
                }

                archiveArtifacts artifacts: "**/logs/**", excludes: null, allowEmptyArchive: true
                fingerprint: true
            }
        }
    } catch (Exception e) {
        echo "Build failed: ${e}"
        currentBuild.result = 'FAILURE'
    } finally {
        sh "docker rm -f ${env.CONTAINER_ID}"
    }
}
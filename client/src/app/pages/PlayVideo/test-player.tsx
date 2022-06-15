/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import { useEffect, useState, useRef } from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';
import { io } from 'socket.io-client';
import {
  LoadingOutlined,
  MenuOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from '@ant-design/icons';
import DataTable from 'react-data-table-component';

import { ReactComponent as HeadIcon } from './assets/head.svg';
import { ReactComponent as ShoulderIcon } from './assets/shoulder.svg';
import { ReactComponent as BowIcon } from './assets/bow.svg';
import { ReactComponent as HandIcon } from './assets/hand.svg';
import { ReactComponent as TorsoIcon } from './assets/torso.svg';
import { ReactComponent as KneeIcon } from './assets/knee.svg';

import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  RadialLinearScale,
  ArcElement,
} from 'chart.js';
import { Line, PolarArea } from 'react-chartjs-2';
import { NavBar } from '../../components/NavBar';
import { useQuery } from '../../../hooks/useQuery';

import { Player } from 'video-react';
import ReactPlayer from 'react-player';
import { getJSON } from 'utils/getJson';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement,
);

export const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: 'Per Frame Stats',
    },
  },
  scales: {
    x: {
      display: true,
      title: {
        display: true,
      },
    },
    y: {
      display: true,
      title: {
        display: true,
        text: 'Value',
      },
      suggestedMin: -30,
      suggestedMax: 30,
    },
  },
  elements: {
    point: {
      radius: 0,
    },
  },
};

export const POLAR_DATA_EXAMPLE = {
  labels: [
    'Head Position',
    'Body Position',
    'Shoulder Position',
    'Right Arm',
    'Left Arm',
    'Bow Position',
  ],
  datasets: [
    {
      label: '# of Votes',
      data: [12, 10, 3, 5, 2, 3],
      backgroundColor: [
        'rgba(255, 99, 132, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 206, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)',
        'rgba(153, 102, 255, 0.5)',
        'rgba(255, 159, 64, 0.5)',
      ],
      borderWidth: 1,
    },
  ],
};

const customStyles = {
  rows: {
    style: {
      height: '24px', // override the row height
    },
  },
  headCells: {
    style: {
      paddingLeft: '8px', // override the cell padding for head cells
      paddingRight: '8px',
      fontWeight: 'bold',
    },
  },
};

const columns = [
  {
    name: 'Name',
    selector: row => row[2],
    style: {
      fontWeight: 'bold',
    },
  },
  {
    name: 'Value',
    selector: row => row[3],
  },
  {
    name: 'Error',
    selector: row => row[4],
    conditionalCellStyles: [
      {
        when: row => row[4] === 'Normal',
        style: {
          backgroundColor: 'rgba(63, 195, 128, 0.9)',
          color: 'white',
          '&:hover': {
            cursor: 'pointer',
          },
        },
      },
      {
        when: row => row[4] !== 'Normal',
        style: {
          backgroundColor: 'rgba(242, 38, 19, 0.9)',
          color: 'white',
          '&:hover': {
            cursor: 'not-allowed',
          },
        },
      },
      // You can also pass a callback to style for additional customization
      {
        when: row => row.calories < 600,
        style: row => ({ backgroundColor: row.isSpecial ? 'pink' : 'inerit' }),
      },
    ],
  },
];

interface Props {}

const OFFSET = 30

export function TestPlayer(props: Props) {
  const query = useQuery();
  // console.log(query.get('title'));
  // @ts-ignore
  const [socket, setSocket] = useState<any>(null);
  const [frame, setFrame] = useState<any>(0);
  const [img, setImg] = useState<string>('');
  const [data, setData] = useState<any>(null);

  const [labels, setLabels] = useState<Array<number>>([0]);

  const [chartData, setChartData] = useState<any>({
    labels: labels,
    datasets: [
      {
        label: 'Head',
        data: [0],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Body',
        data: [0],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Shoulder',
        data: [0],
        borderColor: 'rgba(255,194,44,0.85)',
        backgroundColor: 'rgba(255, 206, 86, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Right Arm',
        data: [0],
        borderColor: 'rgb(75,192,192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Left Arm',
        data: [0],
        borderColor: 'rgb(153,102,255)',
        backgroundColor: 'rgba(153, 102, 255, 0.5)',
      },
      {
        label: 'Bow',
        data: [0],
        borderColor: 'rgb(255,159,64)',
        backgroundColor: 'rgba(255, 159, 64, 0.5)',
      },
      {
        label: 'Knee(s)',
        data: [0],
        borderColor: 'rgb(20,255,150)',
        backgroundColor: 'rgba(20,255,150,0.5)',
      },
    ],
  });

  const [polarData, setPolarData] = useState<any>({
    labels: [
      'Head Position',
      'Body Position',
      'Shoulder Position',
      'Right Arm',
      'Left Arm',
      'Bow Position',
      'Knees Position',
    ],
    datasets: [
      {
        label: '# of Votes',
        // data: [12, 10, 3, 5, 2, 3],
        data: [0, 0, 0, 0, 0, 0, 0],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
          'rgba(255, 159, 64, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  });

  // @ts-ignore
  useEffect(() => {
    setTimeout(() => {
      setSocket(true);
    }, 1000);
  }, []);

  const play = () => {
    setPlaying(true);
  };

  const stop = () => {
    // socket.emit('stop');
    // // eslint-disable-next-line no-restricted-globals
    // location.reload();
  };

  useEffect(() => {
    setTimeout(() => {
      play();
    }, 2000);
  }, []);

  const increment = obj => {
    let headValue = parseInt(obj.data[0][3]);
    let bodyValue = parseInt(obj.data[1][3]);
    let shoulderValue = parseInt(obj.data[2][3]);
    let rightArm = parseInt(obj.data[3][3]);
    let leftArm = parseInt(obj.data[4][3]);
    let bowPosition = parseInt(obj.data[5][3]);
    let kneesPosition = parseInt(obj.data[6][3]);

    let datasetsCopy = chartData.datasets.slice(0);

    let copyHead = datasetsCopy[0].data;
    let copyBody = datasetsCopy[1].data;
    let copyShoulder = datasetsCopy[2].data;
    let copyRightArm = datasetsCopy[3].data;
    let copyLeftArm = datasetsCopy[4].data;
    let copyBow = datasetsCopy[5].data;
    let copyKnees = datasetsCopy[6].data;

    copyHead.push(headValue);
    copyBody.push(bodyValue);
    copyShoulder.push(shoulderValue);
    copyRightArm.push(rightArm);
    copyLeftArm.push(leftArm);
    copyBow.push(bowPosition);
    copyKnees.push(kneesPosition);

    let polarDataCopy = polarData.datasets.slice(0);

    let polarCopy = polarDataCopy[0].data;

    let headStats = obj.data[0][4];
    let bodyStats = obj.data[1][4];
    let shoulderStats = obj.data[2][4];
    let rightArmStats = obj.data[3][4];
    let leftArmStats = obj.data[4][4];
    let bowPositionStats = obj.data[5][4];
    let kneesPositionStats = obj.data[6][4];

    if (headStats !== 'Normal') {
      polarCopy[0] = polarCopy[0] + 1;
    }

    if (bodyStats !== 'Normal') {
      polarCopy[1] = polarCopy[1] + 1;
    }

    if (shoulderStats !== 'Normal') {
      polarCopy[2] = polarCopy[2] + 1;
    }

    if (rightArmStats !== 'Normal') {
      polarCopy[3] = polarCopy[3] + 1;
    }

    if (leftArmStats !== 'Normal') {
      polarCopy[4] = polarCopy[4] + 1;
    }

    if (bowPositionStats !== 'Normal') {
      polarCopy[5] = polarCopy[5] + 1;
    }

    if (kneesPositionStats !== 'Normal') {
      polarCopy[6] = polarCopy[6] + 1;
    }

    setChartData(
      Object.assign({}, chartData, {
        datasets: datasetsCopy,
      }),
    );

    setPolarData(
      Object.assign({}, polarData, {
        datasets: polarDataCopy,
      }),
    );
  };

  const [isShow, setIsShow] = useState<boolean>(false);

  const toggleMenu = () => {
    setIsShow(!isShow);
  };

  const headMessage = (input: string) => {
    switch (input) {
      case 'Head_Normal':
        return '正常';
      case 'E11':
        return '要擺正';
    }
  };

  const bodyMessage = (input: string) => {
    switch (input) {
      case 'Body_Normal':
        return '正常';
      case 'E14':
        return '要坐正';
    }
  };

  const shoulderMessage = (input: string) => {
    switch (input) {
      case 'Shoulder_Normal':
        return '正常';
      case 'E13':
        return '右肩太高';
      case 'E12':
        return '左肩太高';
    }
  };

  const erhuMessage = (input: string) => {
    switch (input) {
      case 'Erhu_Normal':
        return '正常';
      case 'E41':
        return '左傾斜';
      case 'E42':
        return '右傾斜';
    }
  };

  const bowMessage = (input: string) => {
    switch (input) {
      case 'Bow_Normal':
        return '正常';
      case 'E43':
        return '軌跡必須一直線';
      case 'E44':
        return '運弓太高';
      case 'E45':
        return '運弓太低';
    }
  };

  const leftArmMessage = (input: string) => {
    switch (input) {
      case 'LeftHand_Normal':
        return '正常';
      case 'E21':
        return '手臂過高';
      case 'E22':
        return '手臂太低';
      case 'E23':
        return '手腕手肘盡量一直線';
    }
  };

  // const rightArmMessage = (input: string) => {
  //   switch (input) {
  //     case 'RightArm_Normal':
  //       return '正常';
  //     case 'E34':
  //       return '右手腕持弓太向內翻';
  //     case 'E35':
  //       return '右手腕持弓太向外翻';
  //   }
  // };

  const leftHandMessage = (input: string) => {
    switch (input) {
      case 'LeftHand_Normal':
        return '正常';
      case 'A-L1':
        return '錯誤類型1';
      case 'A-L2':
        return '錯誤類型2';
    }
  };


  const rightHandMessage = (input: string) => {
    switch (input) {
      case 'RightHand_Normal':
        return '正常';
      case 'E31':
        return '拇指握弓錯誤';
      case 'E32':
        return '食指觸弓桿錯誤';
      case 'E33':
        return '中指無名指觸弓毛錯誤';
    }
  };

  const kneesMessage = (input: string) => {
    switch (input) {
      case 'Knees_Normal':
        return '正常';
      case 'E15':
        return '要與肩同寬';
    }
  };

  const refVideo = useRef(null);

  // refVideo.current.subscribeToStateChange(state => console.log(state));

  const [jsonData, setJsonData] = useState(null);

  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    getJSON(
      `https://140.115.51.243/api/predict/${query.get('title')}.json`,
      function (err, data) {
        if (err !== null) {
          alert('Something went wrong: ' + err);
        } else {
          console.log(data.length);
          setJsonData(data);
        }
      },
    );
  }, []);

  return (
    <>
      <Helmet>
        <title>{`Evaluate ${query.get('title')}`}</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex h-full w-full bg-black flex-1'}>
          <div
            className={
              'flex flex-col w-full h-full justify-center items-center'
            }
          >
            {/*<div className={'flex flex-row w-full'}>*/}
            {/*  <div*/}
            {/*    className={*/}
            {/*      'flex-1 text-white text-4xl py-4 transition ease-in-out bg-green-500 hover:bg-indigo-700 duration-300 items-center justify-center flex cursor-pointer'*/}
            {/*    }*/}
            {/*    onClick={play}*/}
            {/*  >*/}
            {/*    <PlayCircleOutlined className={'mr-2'} />*/}
            {/*    {'Re-PLAY'}*/}
            {/*  </div>*/}
            {/*  /!*<div*!/*/}
            {/*  /!*  className={*!/*/}
            {/*  /!*    'flex-1 text-white text-4xl py-4 transition ease-in-out bg-red-500 hover:bg-pink-700 duration-300 items-center justify-center flex cursor-pointer'*!/*/}
            {/*  /!*  }*!/*/}
            {/*  /!*  onClick={stop}*!/*/}
            {/*  /!*>*!/*/}
            {/*  /!*  <StopOutlined className={'mr-2'} />*!/*/}
            {/*  /!*  {'STOP'}*!/*/}
            {/*  /!*</div>*!/*/}
            {/*  <div*/}
            {/*    className={*/}
            {/*      'text-white text-4xl py-4 transition ease-in-out bg-blue-500 hover:bg-pink-700 duration-300 items-center justify-center flex cursor-pointer px-8'*/}
            {/*    }*/}
            {/*    onClick={toggleMenu}*/}
            {/*  >*/}
            {/*    <MenuOutlined className={'mr-2'} />*/}
            {/*    {'ANALYZE'}*/}
            {/*  </div>*/}
            {/*</div>*/}
            {socket ? (
              <div className="flex flex-1 justify-center max-w-7xl">
                {/*<div*/}
                {/*  className={*/}
                {/*    'text-white text-3xl absolute mt-4 ml-4 transition ease-in-out bg-black p-2 z-50'*/}
                {/*  }*/}
                {/*>*/}
                {/*  {`Frame Number : ${frame}`}*/}
                {/*</div>*/}

                <div
                  className={'bg-black'}
                  style={{
                    position: 'relative',
                    paddingLeft: 12,
                    minWidth: 960,
                  }}
                >
                  <ReactPlayer
                    className="absolute top-0 left-0"
                    url={`https://140.115.51.243:5001/predict/${query.get(
                      'title',
                    )}_stream.mov`}
                    controls={true}
                    onProgress={state => {
                      const frameNum = Math.round(state.playedSeconds * 30);
                      // console.log(jsonData.length);
                      // console.log(frameNum);
                      if (frameNum < jsonData.length) {
                        setFrame(Math.round(frameNum) + OFFSET);
                      }
                    }}
                    // onPlay={() => console.log('play')}
                    progressInterval={1}
                    width="100%"
                    height="100%"
                    playbackRate={0.5}
                    playing={playing}
                    muted={true}
                  />
                </div>

                {jsonData ? (
                  <div
                    className={
                      'flex flex-col text-white text-lg absolute mt-4 ml-4 transition ease-in-out p-2 bottom-0 left-0 mb-2 z-10 mb-24'
                    }
                    style={{
                      minWidth: 300,
                      backgroundColor: 'rgba(0,47,105,0.35)',
                    }}
                  >
                    {/*HEAD*/}
                    <div
                      className={
                        jsonData[frame][0][0][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-orange-500 mb-2'
                      }
                    >
                      <HeadIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {`頭 : ${headMessage(jsonData[frame][0][0][0])}`}
                      </div>
                    </div>

                    {/*TORSO*/}
                    <div
                      className={
                        jsonData[frame][0][1][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-orange-500 mb-2'
                      }
                    >
                      <TorsoIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {`坐姿 : ${bodyMessage(jsonData[frame][0][1][0])}`}
                      </div>
                    </div>

                    {/*SHOULDER*/}
                    <div
                      className={
                        jsonData[frame][0][2][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-orange-500 mb-2'
                      }
                    >
                      <ShoulderIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {`肩 : ${shoulderMessage(jsonData[frame][0][2][0])}`}
                      </div>
                    </div>

                    {/*ERHU MESSAGE*/}
                    <div
                      className={
                        jsonData[frame][0][3][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-yellow-500 mb-2'
                      }
                    >
                      <BowIcon className={'w-12 h-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {/*{`Left : ${data[4][0]}`}*/}
                        {`琴桿 : ${erhuMessage(jsonData[frame][0][3][0])}`}
                      </div>
                    </div>

                    {/*BOW MESSAGE*/}
                    <div
                      className={
                        jsonData[frame][0][4][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-green-500 mb-2'
                      }
                    >
                      <BowIcon className={'w-12 h-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {/*{`Bow : ${data[5][0]}`}*/}
                        {`運弓 : ${bowMessage(jsonData[frame][0][4][0])}`}
                      </div>
                    </div>

                    {/*LEFT HAND*/}
                    <div
                      className={
                        jsonData[frame][0][5][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-red-500 mb-2'
                      }
                    >
                      <HandIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {/*{`Left : ${data[4][0]}`}*/}
                        {`左手持琴 : ${leftArmMessage(jsonData[frame][0][5][0])}`}
                      </div>
                    </div>

                    {/*LEFT HAND FRET*/}
                    <div
                      className={
                        'flex items-center mb-2'
                      }
                    >
                      <HandIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {/*{`Left : ${data[4][0]}`}*/}
                        {`左手按弦 : 正常`}
                      </div>
                    </div>

                    {/*RIGHT HAND*/}
                    <div
                      className={
                        jsonData[frame][0][6][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-red-500 mb-2'
                      }
                    >
                      <HandIcon className={'w-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {`右手持弓 : ${rightHandMessage(jsonData[frame][0][6][0])}`}
                      </div>
                    </div>

                    {/*/!*RIGHT ARM*!/*/}
                    {/*<div*/}
                    {/*  className={*/}
                    {/*    jsonData[frame][0][4][4] === 'Normal'*/}
                    {/*      ? 'flex items-center mb-2'*/}
                    {/*      : 'flex items-center text-red-500 mb-2'*/}
                    {/*  }*/}
                    {/*>*/}
                    {/*  <ShoulderIcon className={'w-12 mr-4'} />*/}
                    {/*  <div className={'whitespace-nowrap'}>*/}
                    {/*    {`右臂 : ${rightArmMessage(jsonData[frame][0][4][0])}`}*/}
                    {/*  </div>*/}
                    {/*</div>*/}

                    {/*KNEES*/}
                    <div
                      className={
                        jsonData[frame][0][8][4] === 'Normal'
                          ? 'flex items-center mb-2'
                          : 'flex items-center text-orange-500 mb-2'
                      }
                    >
                      <KneeIcon className={'w-12 h-12 mr-4'} />
                      <div className={'whitespace-nowrap'}>
                        {/*{`Knee(s) : ${kneesMessage(data[6][0])}`}*/}
                        {`兩膝 : ${kneesMessage(jsonData[frame][0][8][0])}`}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <div>Not Connected</div>
            )}
          </div>
        </div>
        {isShow ? (
          <div
            style={{ flex: 0.7 }}
            className={'h-full flex flex-col bg-white overflow-scroll'}
          >
            {data ? (
              <div className={'h-full mb-8'}>
                <DataTable
                  columns={columns}
                  data={data}
                  customStyles={customStyles}
                  dense={true}
                />
              </div>
            ) : null}
            <div className={'h-full mb-4'}>
              <Line options={options} data={chartData} />
            </div>
            <div className={'px-12 mb-24'}>
              <PolarArea
                data={polarData}
                options={{
                  plugins: {
                    legend: {
                      position: 'top' as const,
                    },
                    title: {
                      display: true,
                      text: 'Error Plot',
                    },
                  },
                }}
              />
            </div>
          </div>
        ) : null}
      </PageWrapperMain>
    </>
  );
}

const Wrapper = styled.header`
  box-shadow: 0 1px 0 0 ${p => p.theme.borderLight};
  height: ${StyleConstants.NAV_BAR_HEIGHT};
  display: flex;
  position: fixed;
  top: 0;
  width: 100%;
  background-color: ${p => p.theme.background};
  z-index: 2;

  @supports (backdrop-filter: blur(10px)) {
    backdrop-filter: blur(10px);
    background-color: ${p =>
      p.theme.background.replace(
        /rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/,
        'rgba$1,0.75)',
      )};
  }

  ${PageWrapper} {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;

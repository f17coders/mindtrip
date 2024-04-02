// 멤버에 들어갈 타입
type memberType = {
  memberId: number|null,
  nickname: string|null,
  socialId: string|null,
  villageId: number|null,
  villageName: 'apple'|'orange'|'pineapple'|'watermelon'|'grape'|'peach'|'blueberry'|'kakao',
  level: number,
  missionCount: number,
  accessToken :string|null,
} 

// 마을 타입
type villageType = {
  apple : string,
  orange : string,
  pineapple : string,
  watermelon : string,
  grape : string,
  peach : string,
  blueberry : string,
  kakao: string
}

// 가능한 마을의 타입
type villageNameType = 'apple'|'orange'|'pineapple'|'watermelon'|'grape'|'peach'|'blueberry'|'kakao'


// 고민 타입
type consultType = {
  consultId : number,
  memberId : number,
  title : string,
  content : string,
  categoryId : number,
  canLike: boolean,
  likeCount: number,
  channelId: string|null,
  nickname: string,
  isClosed : boolean
}

// 고민 카테고리 타입
type categoryType = {
  categoryId : number,
  categoryName : string
}

// 채팅방 타입
type chattingRoom = {
  consultId: number,
  memberId: number,
  nickname: string,
  title:string,
  channelId: string,
  text: string|null,
  shared: boolean
}


export type {consultType, categoryType, memberType, villageType, chattingRoom, villageNameType}